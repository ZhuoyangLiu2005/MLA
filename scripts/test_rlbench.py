import os, sys, pathlib
import argparse
import tqdm
import shutil
import logging
import time
from datetime import datetime
from termcolor import cprint, colored

from lift3d.envs.rlbench_env import RLBenchEnv, RLBenchActionMode, RLBenchObservationConfig
from lift3d.helpers.gymnasium import VideoWrapper
from lift3d.helpers.common import Logger
from lift3d.helpers.graphics import EEpose

import numpy as np
import pickle

import os
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

from load_mla import load_vla
import torch
from PIL import Image



def setup_logger(log_dir):
    log_filename = os.path.join(log_dir, "output.log")

    logger = logging.getLogger("RLBenchLogger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def recreate_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)

def model_load(args):
    model = load_vla(
            args.model_path,
            load_for_training=False,
            action_model_type='DiT-B',
            future_action_window_size=int(args.model_action_steps),
            load_dit = False,
            hf_token=args.hf_token,
            use_diff = args.use_diff,
            use_pointcloud = args.use_pointcloud,
        )
    # model.vlm = model.vlm.to(torch.bfloat16)
    model.to(f'cuda:{args.cuda}').eval()
    return model

def model_predict(args, model, image, point_cloud, prompt, cur_robot_state=None):
    actions_diff = model.predict_action_diff(
        image = image,
        pointcloud = point_cloud,
        instruction = prompt,
        unnorm_key='rlbench',
        cur_robot_state = cur_robot_state,
    )
    return actions_diff

def cal_cos(a,b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_similarity = dot_product / (norm_a * norm_b + 1e-7)
    return cosine_similarity

def main(args):
    # Report the arguments
    Logger.log_info(f'Running {colored(__file__, "red")} with arguments:')
    Logger.log_info(f'task name: {args.task_name}')
    Logger.log_info(f'number of episodes: {args.num_episodes}')
    Logger.log_info(f'result directory: {args.result_dir}')
    Logger.log_info(f'exp name: {args.exp_name}')
    Logger.log_info(f'actions steps: {args.model_action_steps}')
    Logger.log_info(f'max steps: {args.max_steps}')
    Logger.log_info(f'cuda used: {args.cuda}')
    cprint('-' * os.get_terminal_size().columns, 'cyan')

    action_mode = RLBenchActionMode.eepose_then_gripper_action_mode(absolute=True)
    obs_config = RLBenchObservationConfig.single_view_config(camera_name='front', image_size=(224, 224))
    env = RLBenchEnv(
        task_name=args.task_name,
        action_mode=action_mode,
        obs_config=obs_config,
        point_cloud_camera_names=['front'],
        cinematic_record_enabled=True,
        num_points=1024,
        use_point_crop=True
    )
    env = VideoWrapper(env)
    args.result_dir = os.path.join(args.result_dir, 'predict_results')
    
    if args.exp_name is None:
        args.exp_name = args.task_name

    video_dir = os.path.join(
        args.result_dir, args.task_name, args.exp_name, "videos"
    )
    recreate_directory(video_dir)
    log_dir = os.path.join(video_dir, f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    recreate_directory(log_dir)
    logger = setup_logger(log_dir)
    
    success_num = 0

    model = model_load(args)
    episode_length = args.max_steps

    num_episodes = args.num_episodes
    success_list = np.zeros(num_episodes) 
    for i in range(num_episodes):
        
        logger.info(f'episode: {i}, steps: {episode_length}')
        obs_dict = env.reset()
        terminated = False
        rewards = 0
        success = False
        ar_sum = 0 
        ar_cnt = 0
        ar_time_sum = 0
        diff_time_sum =0

        cur_robot_state = np.array([ 0.27849028, -0.00815899, 1.47193933, -3.14159094,  0.24234043,  3.14158629,  1.        ])
        for j in range(episode_length):

            front_image = obs_dict['image']
            front_image = Image.fromarray(front_image)
            robot_state = obs_dict['robot_state']
            point_cloud = obs_dict['point_cloud']
            point_cloud = torch.tensor(point_cloud).to(torch.float).unsqueeze(0)
            print(point_cloud.shape)
            prompt = env.text
            if args.use_robot_state:
                action_diff = model_predict(args, model, front_image, point_cloud, prompt, cur_robot_state)
            else:
                action_diff = model_predict(args, model, front_image, point_cloud, prompt)
            
            action = action_diff[0]
            logger.info("only use diff_action!")
            
            action[:3] += robot_state[7:10]
            cur_robot_state = action
            gripper_open = action[-1]
            action = EEpose.pose_6DoF_to_7DoF(action[:-1])
            action = np.append(action, gripper_open)
            logger.info("%d  : %s", j, action)


            obs_dict, reward, terminated, truncated, info = env.step(action)
            ar_cnt += 1
            rewards += reward
            success = success or bool(reward)

            if terminated or truncated or success:
                break
        
        logger.info("average ar_sum = %f", ar_sum / ar_cnt)
        logger.info("average ar infer time = %f", ar_time_sum / ar_cnt)
        logger.info("average diff infer time = %f", diff_time_sum / ar_cnt)
        if success:
            success_num += 1
            success_list[i] = 1

        image_dir = os.path.join(
            args.result_dir, args.task_name, args.exp_name, "images", f"episode{i}"
        )
        recreate_directory(image_dir)
        depth_dir = os.path.join(
            args.result_dir, args.task_name, args.exp_name, "depths", f"episode{i}"
        )
        recreate_directory(depth_dir)

        env.save_video(os.path.join(video_dir, f'episode{i}_video_steps.mp4'))
        env.save_images(image_dir, quiet=True)
        env.save_depths(depth_dir, quiet=True)
        logger.info(f'video saved to {video_dir}')
        logger.info(f'episode{i}_{success}')
        Logger.print_seperator()
    
    logger.info(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_num/args.num_episodes*100}%')
    with open(os.path.join(args.result_dir, args.task_name, f'{args.exp_name}_success_rate.txt'), "w", encoding="utf-8") as file:
        file.write(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_num/args.num_episodes*100}%')
    logger.info(f"Success list: {success_list}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # exp parameters
    parser.add_argument('--cuda', type=str, default='7')
    parser.add_argument('--result_dir', type=str, default='s')
    parser.add_argument('--exp_name', type=str, default=None)
    # env parameters
    parser.add_argument('--task_name', type=str, default='close_box')
    parser.add_argument('--num_episodes', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=10)
    # model parameters
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model_action_steps', type=str, default='15')
    parser.add_argument('--cfg_scale', type=str, default='1.5')
    parser.add_argument('--use_diff', type=int, default=0)
    parser.add_argument('--use_pointcloud', type=int, default=0)
    parser.add_argument('--use_wrist', type=int, default=0)
    parser.add_argument('--hf_token', type=str, default='')
    parser.add_argument('--use_robot_state', type=int, default=0)
    main(parser.parse_args())