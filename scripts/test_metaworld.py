import argparse
import copy
import os
import pathlib
import sys
from PIL import Image
import pickle
import numpy as np
import tqdm
from datetime import datetime
from termcolor import colored, cprint
from load import load_vla
from lift3d.envs import METAWORLD_LANGUAGE_DESCRIPTION, MetaWorldEnv, load_mw_policy
from lift3d.helpers.common import (
    Logger,
    save_point_cloud_ply,
    save_rgb_image,
    save_video_imageio,
    save_npy_file,
)
import logging
import shutil
from lift3d.helpers.common import Logger, set_seed

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
            future_action_window_size=0,
            load_dit = False,
            hf_token=args.hf_token,
            use_diff = args.use_diff,
            use_pointcloud = args.use_pointcloud,
            action_dim = args.action_dim,
        )
    # (Optional) use "model.vlm = model.vlm.to(torch.bfloat16)" to load vlm in bf16
    model.to(f'cuda:{args.cuda}').eval()
    return model

def model_predict(model, image, prompt, cur_robot_state=None, point_cloud=None):
    actions_diff = model.predict_action_diff(
            image = image,
            pointcloud = point_cloud,
            instruction = prompt,
            unnorm_key='metaworld',
            cur_robot_state = cur_robot_state,
            action_dim = 4,
        )
    return [actions_diff]

def main(args):
    # Report the arguments
    # set_seed(0)
    Logger.log_info(
        f'Running {colored(pathlib.Path(__file__).absolute(), "red")} with arguments:'
    )
    Logger.log_info(f"Task name: {args.task_name}")
    Logger.log_info(f"Camera name: {args.camera_name}")
    Logger.log_info(f"Image size: {args.image_size}")
    Logger.log_info(f"Number of episodes: {args.num_episodes}")
    Logger.log_info(f"Episode max length: {args.episode_max_length}")
    Logger.log_info(f"Save directory: {args.save_dir}")
    Logger.log_info(f"Use diff: {args.use_diff}")
    Logger.print_seperator()

    if args.replay_or_predict == 'predict':
        model = model_load(args)

    task_name = args.task_name

    # Create the save directory
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    video_dir = (
        pathlib.Path(args.save_dir)
        / "visualized_data"
        / "videos"
        / task_name
        / args.camera_name
    )
    video_dir.mkdir(parents=True, exist_ok=True)

    log_dir = os.path.join(video_dir, f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    recreate_directory(log_dir)
    logger = setup_logger(log_dir)

    env = MetaWorldEnv(
        task_name=task_name,
        image_size=args.image_size,
        camera_name=args.camera_name,
        point_cloud_camera_names=[
            args.camera_name,
        ],
    )
    original_episode_length = env.max_episode_length
    env.max_episode_length = args.episode_max_length
    Logger.log_info(
        f"Original max episode length: {original_episode_length}, New max episode length: {env.max_episode_length}"
    )

    total_success = 0

    description = METAWORLD_LANGUAGE_DESCRIPTION[task_name]
    episode_idx = 0
    while episode_idx < args.num_episodes:
        obs_dict = env.reset()
        truncated = False
        terminated = False
        ep_reward = 0.0
        ep_success = False
        ep_success_times = 0
        img_arrays_sub = []

        logger.info(f'episode: {episode_idx}, steps: {args.episode_max_length}')

        step = 0
        while not truncated and not terminated:

            raw_state = obs_dict["raw_state"]
            obs_img = obs_dict["image"]
            obs_robot_state = obs_dict["robot_state"]
            obs_point_cloud = obs_dict["point_cloud"] if args.use_pointcloud else None
            obs_point_cloud = obs_point_cloud[:,:3]
            obs_point_cloud = np.expand_dims(obs_point_cloud, axis=0)
            img_arrays_sub.append(obs_img)
            obs_img = Image.fromarray(obs_img)

            actions_diff = model_predict(model, obs_img, description, obs_robot_state, obs_point_cloud)
            action = actions_diff[0][0]

            # logger.info("%d  : %s", step, action)
            obs_dict, reward, terminated, truncated, env_info = env.step(action)
            ep_reward += reward

            ep_success = ep_success or env_info["success"]
            ep_success_times += env_info["success"]
            step += 1
            if truncated or ep_success_times:
                break

        img_arrays_sub.append(obs_dict["image"])

        logger.info(f'episode{episode_idx}_{ep_success}')

        sample_video_array = np.stack(img_arrays_sub, axis=0)
        save_video_imageio(
            sample_video_array,
            video_dir / f"episode_{episode_idx}.mp4",
        )

        total_success += ep_success
        episode_idx += 1

    logger.info(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {total_success/args.num_episodes*100}%')
    with open(os.path.join(args.save_dir, args.task_name, f'{args.exp_name}_success_rate.txt'), "w", encoding="utf-8") as file:
        file.write(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {total_success/args.num_episodes*100}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # env parameters
    parser.add_argument("--task_name", type=str, default="assembly")
    parser.add_argument("--camera_name", type=str, default="corner")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_episodes", type=int, default=30)
    parser.add_argument("--episode_max_length", type=int, default=200)
    parser.add_argument('--replay_or_predict', type=str, default='predict')
    # device parameters
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(
            pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "metaworld"
        ),
    )
    # model parameters
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--hf_token', type=str, default='')
    parser.add_argument('--model_action_steps', type=str, default='0')
    parser.add_argument('--action_dim', type=int, default=4)
    parser.add_argument('--use_diff', type=int, default=1)
    parser.add_argument('--use_pointcloud', type=int, default=1)
    parser.add_argument('--use_robot_state', type=int, default=1)
    parser.add_argument('--cfg_scale', type=str, default='0.0')
    main(parser.parse_args())
