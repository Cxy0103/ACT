
from PIL import Image
from io import BytesIO
import torch, gc
import sys
import json
import time
import os
import pickle
import numpy as np
import scipy.spatial.transform as st
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import time
import argparse

from scipy.spatial.transform import Rotation as R

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
from robomimic.utils.file_utils import get_env_metadata_from_dataset
from robomimic.utils.env_utils    import create_env_for_data_processing, create_env_from_metadata
import h5py
import cv2
import tempfile
import imageio
import inspect
import random

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

gc.collect()
torch.cuda.empty_cache()

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
SEED = 42

'''robosuite部分'''

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_env_from_hdf5(hdf5_path, camera="agentview", version="v1.2.0", control_delta=False, wrist_camera=True):
    """
    从 hdf5 文件中读取 env_args，然后调用 robosuite.make() 构造环境。

    Args:
        hdf5_path (str): 指向包含 env_args 的 HDF5 文件路径。
        camera (str): 要保留的相机视角名称（如 "agentview" 或其他）。
    Returns:
        env (robosuite.environments.RobosuiteEnv): 构建好的环境实例。
    """
    # 1. 读取并解析 JSON 格式的 env_args
    with h5py.File(hdf5_path, "r") as f:
        raw = f["/data"].attrs["env_args"]
        env_args = json.loads(raw)
        xml_str = f["data/demo_1"].attrs["model_file"]

    # 2. 写成一个临时 XML 文件
    # tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    # xml_path = tmp.name
    # tmp.write(xml_str)
    # tmp.close()

    old_asset_path = "/home/robot/installed_libraries/robosuite/robosuite/models/assets/"

    # 4. Get the correct, new asset path on your CURRENT machine
    new_asset_path = suite.models.assets_root  + "/"

    # 5. Replace the old path with the new path in the XML string
    xml_str_repaired = xml_str.replace(old_asset_path, new_asset_path)


    # 2. 从 env_args 里提取最顶层 env_name 和 env_kwargs
    env_name = env_args["env_name"]
    kwargs    = env_args["env_kwargs"].copy()

    print("env_args:", env_args)
    # print("env_kwargs:", env_args["env_kwargs"])
    print("kwargs:", kwargs)

    if version == "v1.4.0":
        kwargs["has_renderer"] = True
    elif version == "v1.2.0":
        kwargs["has_renderer"] = False


    kwargs["has_offscreen_renderer"] = True



    #绝对 or delta
    if control_delta:
        kwargs["controller_configs"]['control_delta'] = True
    else:
        kwargs["controller_configs"]['control_delta'] = False


    kwargs["ignore_done"] = False 

    # 3. 取出 robots 列表 和 controller_configs（单一机器人场景下通常只有一个）
    robots = kwargs.pop("robots")
    controller_cfg = kwargs.pop("controller_configs")
    print(f"robots: {robots}")
    print(f"controller_cfg: {controller_cfg}")

    # 5. 只保留用户指定的单一 camera 视角，并更新对应的 heights/widths
    all_cams       = kwargs.pop("camera_names")
    all_heights    = kwargs.pop("camera_heights")
    all_widths     = kwargs.pop("camera_widths")
    #all_heights    = 224
    #all_widths     = 224
    all_depths     = kwargs.pop("camera_depths")
    try:
        idx = all_cams.index(camera)
    except ValueError:
        raise ValueError(f"Camera '{camera}' 不在 env_args 里: {all_cams}")
    
    
    print(f"idx: {idx}, all_heights: {all_heights}, all_widths: {all_widths}, all_depths: {all_depths}")

    # 构造只包含一个视角的参数
    kwargs["camera_names"]   = [camera]
    kwargs["camera_heights"] = [all_heights]
    kwargs["camera_widths"]  = [all_widths]
    kwargs["camera_depths"]  = [all_depths] if isinstance(all_depths, bool) else [all_depths[idx]]

    if wrist_camera:
        kwargs["camera_names"].append("robot0_eye_in_hand")
        kwargs["camera_heights"].append(all_heights)
        kwargs["camera_widths"].append(all_widths)
        kwargs["camera_depths"].append(all_depths) if isinstance(all_depths, bool) else [all_depths[idx]]

    

    print("modified kwargs:", kwargs)

    # 6. 调用 robosuite.make，传入解析后的所有 kwarg
    env = suite.make(
        env_name=env_name,
        robots=robots,
        # mujoco_xml=xml_path,
        controller_configs=controller_cfg,
        **kwargs
    )
    return env, xml_str_repaired

'''VLA部分'''



def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


def get_robosuite_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = img[::-1, :]

    wrist_img = obs["robot0_eye_in_hand_image"]
    wrist_img = wrist_img[::-1, :]
    
    
    # img = rearrange(img, 'h w c -> c h w')
    # img = np.stack([img], axis=0)  # Add batch dimension
    # img = torch.from_numpy(img / 255.0).float().cuda().unsqueeze(0)

    return img, wrist_img

def get_image(obs, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(obs[cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def get_ACT_img(img):
    """Extracts image from observations and preprocesses it."""
    img = rearrange(img, 'h w c -> c h w')
    img = np.stack([img], axis=0)  # Add batch dimension
    img = torch.from_numpy(img / 255.0).float().cuda().unsqueeze(0)
    return img

def get_libero_dummy_action(model_family: str, obs, control_delta=False):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    # for key in obs:
    #     print(f"{key}: {obs[key].shape}, {obs[key].dtype}")
    # return [0, 0, 0, 0, 0, 0, -1]


    # delta
    # return [0, 0, 0, 0, 0, 0, -1]

    # 绝对坐标
    if control_delta:
        noop_action = [0, 0, 0, 0, 0, 0, -1]
    else:
        pos = obs['robot0_eef_pos']
        
        # 将欧拉角硬编码为 [0, 0, 0]
        # 这样可以测试位置控制是否准确，而不受姿态转换错误的影响
        euler_angles = np.array([0.0, 0.0, 0.0]) # <--- 关键修改
        
        gripper_state = -1.0
        noop_action = np.concatenate([pos, euler_angles, [gripper_state]])
        
    return noop_action
    


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE_TIME}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        #增加分辨率
        img = cv2.resize(img, (256, 256))  # 调整为224x224分辨率
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def initialize_model(config, ckpt_path):

    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

def main(args):

    # 时间集成
    USE_TEMPORAL_AGG = False # <--- 在这里切换 True / False 来对比效果
    
    ckpt_dir = args['ckpt_dir']
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')

    #--------------------------初始化ACT--------------------------#
    print("正在初始化模型...")
    #processor, vla, action_head = initialize_model(cfg)
    config = {
        'ckpt_dir': ckpt_dir,
        'state_dim': 7,
        'real_robot': False,
        'policy_class': 'ACT',  # 或 'CNNMLP'
        'onscreen_render': False,
        'policy_config': {'lr': 1e-5,
                         'num_queries': 20,
                         'kl_weight': 10,
                         'hidden_dim': 512,
                         'dim_feedforward': 3200,
                         'lr_backbone': 1e-5,
                         'backbone': 'resnet18',
                         'enc_layers': 4,
                         'dec_layers': 7,
                         'nheads': 8,
                         'camera_names': ['agentview'],
                         },
        'camera_names': ['agentview'],
        'episode_len': 200,
        'task_name': 'sim_lift',
        'seed': 0,  # 随机种子
        'num_epoches': 2000,  # 训练轮数
        'temporal_agg': USE_TEMPORAL_AGG,  # 是否使用时间集成
        'onscreen_cam': 'angle',  # 仅在 onscreen_render 为 True
    }
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    
    print("fuck1")
    policy = make_policy(policy_class, policy_config)
    print("fuck2")
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']


    #--------------------------初始化ACT完成--------------------------#

    

    # 初始化随机种子
    set_seed_everywhere(SEED)

    # 初始化rbosuite环境
    HDF5_PATH = "/media/hdd1/cs24-chenxy/robomimic_dataset/PickPlaceCan_abs.hdf5"
    prompt = "can"
    version = "v1.2.0"  # 可以是 "v1.2.0" 或 "v1.4.0"
    control_delta = False  # 是否使用 delta 控制
    env, xml_str = make_env_from_hdf5(HDF5_PATH, camera="agentview", version=version, control_delta=control_delta, wrist_camera=True)

    max_steps = 200
    task_successes = 0
    total_task_cnt = 50
    total_episodes = 0

    # 根据是否使用时间集成，设置查询频率
    query_frequency = config['policy_config']['num_queries']
    if config['temporal_agg']:
        query_frequency = 1
        num_queries = config['policy_config']['num_queries']
        print("\n已启用时间集成 (Temporal Aggregation)。")
    else:
        print("\n已禁用时间集成，将使用朴素的动作分块执行。")


    for i in range(total_task_cnt):
        obs = env.reset()
        dummy_t = 0
        replay_images = []

        # ==================> 时间集成所需的核心数据结构 <==================
        if config['temporal_agg']:
            # 创建一个巨大的“档案柜”，用于存储所有历史计划
            all_time_actions = torch.zeros([max_steps, max_steps + num_queries, config['state_dim']]).cuda()

        

        ## 这里是一个示范，先让机器人执行一些空动作，等待环境稳定
        while dummy_t < 2:
            obs, reward, done, info = env.step(get_libero_dummy_action("openvla", obs, control_delta=control_delta))
            dummy_t += 1
            time.sleep(1)  # 等待5秒，模拟机器人执行动作的时间
            


        ## 开始执行任务
        
        print(f"Episode {i+1}/{total_task_cnt}")

        

        for t in tqdm(range(max_steps)):
            image, wrist_image = get_robosuite_image(obs, resize_size=(84, 84))
            replay_images.append(image)

            image = get_ACT_img(image)
            wrist_image = get_ACT_img(wrist_image)


            cos_q = obs.pop("robot0_joint_pos_cos")   # (7,)
            sin_q = obs.pop("robot0_joint_pos_sin")   # (7,)
            qpos = np.arctan2(sin_q, cos_q)           # (7,) 原始关节角度，-π..π
            qpos = pre_process(qpos)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            #---------------------------------------新方法--------------------------------------------------------#

            
            # --- 核心动作生成逻辑 ---
            with torch.inference_mode():
                if config['temporal_agg']:
                    # --- 时间集成模式 ---
                    # 1. 每一步都获取一个新的动作计划
                    all_actions = policy(qpos, image)

                    # 2. 将新计划存入“档案柜”
                    all_time_actions[[t], t:t + num_queries] = all_actions

                    # 3. 收集所有对当前步t的预测
                    actions_for_curr_step = all_time_actions[:, t]

                    # 4. 剔除无效的（全为0的）预测
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]

                    # 5. 计算指数衰减权重 (越早的计划权重越高)
                    k = 0.5
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)

                    # 6. 加权平均得到最终动作
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

                else:
                    # --- 朴素分块模式 ---
                    # 1. 每隔 query_frequency 步才获取一次新计划
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, image)
                    
                    # 2. 从当前计划中，按顺序提取这一步要执行的动作
                    raw_action = all_actions[:, t % query_frequency]
            
            #对动作进行后处理
            action = raw_action.squeeze(0).detach().cpu().numpy()
            action = post_process(action)
            action = normalize_gripper_action(action, binarize=True)

            print(f"step: {t}, action: {action}")
                
            obs, reward, done, info = env.step(action.tolist())
            
            

            if env._check_success():
                print(f"Task success at step {t}!")
                task_successes += 1
                break
            

            #---------------------------------------新方法--------------------------------------------------------#


            #---------------------------------------老方法--------------------------------------------------------#
            '''
            
           
            actions = policy(qpos, image)
            actions = actions.squeeze(0)  # (1, 20, 7) -> (20, 7)
            #actions = actions.cpu().detach().numpy()  # 转换为 NumPy 数组
            # print(f"actions: {actions}")
            # print(f"actions.shape: {actions.shape}, actions.dtype: {actions.dtype}")


            for action in actions:
                action = action.squeeze(0).detach().cpu().numpy()
                action = post_process(action)
                action = normalize_gripper_action(action, binarize=True)
                
                
                print(f"step: {t}, action: {action}")
                
                
                obs, reward, done, info = env.step(action.tolist())
                
                image_for_video = get_robosuite_image(obs, resize_size=(84, 84))
                replay_images.append(image_for_video)

                if env._check_success():
                    task_successes += 1
                    break

                if version == "v1.4.0":
                    env.render()


            if env._check_success():
                print(f"Task success at step {t}!")
                break
            '''
            
            #---------------------------------------老方法--------------------------------------------------------#

        
            
        total_episodes += 1
        print(f"Episode {i+1}/{total_task_cnt} finished. Steps taken: {t}, Task success: {env._check_success()}")
        save_rollout_video(
            replay_images, total_episodes, success=env._check_success(), task_description=prompt, log_file=None
        )
    print(f"Total task successes: {task_successes}/{total_task_cnt}, task success rate: {task_successes / total_task_cnt:.2f}")
    
    '''
    while True:
        

        # 图片路径
        image_path = "/home/user/Project/openvla_umi/openvla_sample/50_frame.png"  # 替换为你的图片路径

        # 处理图像
        print("正在处理图像...")
        image = Image.open(image_path)
        
            
        
        # 生成动作
        print("正在生成机器人动作...")
        action = generate_robot_action(processor, vla, image)
        print(f"action.shape: {action.shape}")
        print(f"action.type: {type(action)}")
        
        if action is not None:
            print("动作生成完成！")
            
            print(f"action is: {action}")

            # with open("action.json", 'w') as f:
            #     json.dump(action.tolist() + [id], f)
            #     id += 1

            print(f"生成的动作向量: {action}")

        #remote_controller.send_action()
        time.sleep(1)
    '''
    
        

# ==================> 核心修改在这里 <==================
if __name__ == "__main__":
    # 1. 创建一个参数解析器
    parser = argparse.ArgumentParser()

    # 2. 定义所有你想从命令行控制的参数
    #    这部分应该和 imitate_episodes.py 中的参数保持一致
    parser.add_argument('--policy_class', action='store', type=str, help='Policy class, capitalize.', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='Checkpoint directory.', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--seed', action='store', type=int, help='Seed.', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='Number of epochs.', required=True)
    
    # 您可以根据需要添加更多参数，比如 chunk_size, dec_layers 等
    # parser.add_argument('--chunk_size', action='store', type=int, help='Chunk size for ACT.', default=100)
    # parser.add_argument('--dec_layers', action='store', type=int, help='Number of decoder layers.', default=1)

    # 3. 解析命令行传入的参数
    #    vars() 将解析后的参数对象转换为一个字典
    main(vars(parser.parse_args()))
