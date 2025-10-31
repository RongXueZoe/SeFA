"""
For loading data from PKL files in MetaWorld dataset.
"""

import numpy as np
from typing import Dict
import cv2
import os
import pickle
import time
from PIL import Image
from tqdm import tqdm
import torch

def resize_img(img, size=(96, 96)):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    im_pil = Image.fromarray(img)
    im_pil = im_pil.resize(size, Image.ANTIALIAS)
    img = np.array(im_pil)
    return np.transpose(img, (2, 0, 1))

class ReplayBufferPKL:
    def __init__(self, buffer_size: int, video_shape: tuple, fps: float = 30.0):
        self.buffer_size = buffer_size
        self.video_shape = video_shape
        self.fps = fps
        self.paths = []
        self.imgs = []

    @property
    def backend(self):
        return 'numpy'

    @classmethod
    def load_from_path(cls, path: str, buffer_size: int, video_shape: tuple, fps: float = 30.0, split: str = "train"):
        buffer = cls(buffer_size, video_shape, fps)

        # Find all MP4 and PKL files in the directory
        files = os.listdir(path)
        pkl_files = [f for f in files if f.endswith('.pkl') and not f.startswith('task') and 'seed' in f]
        if split == 'train':
            pkl_files = pkl_files[:45]
        elif split == 'val':
            pkl_files = pkl_files[45:50]

        # Load episode data
        for pkl_file in pkl_files:
            with open(os.path.join(path, pkl_file), "rb") as f:
                path_data = pickle.load(f)
                path_data["path_length"] = path_data["actions"].shape[0]
                buffer.paths.append(path_data)
                video_path = path_data['video_path'].replace('.mp4', '_96.npz')
                try:
                    frames = np.load(video_path)['arr_0']
                except:
                    # if frames are not saved in npz format, load mp4 and save frames
                    video_path = path_data['video_path']
                    # Load video data
                    cap = cv2.VideoCapture(video_path)
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = resize_img(frame)
                        frames.append(frame)
                    cap.release()
            buffer.imgs.append(frames)

        buffer.num_tasks = len(buffer.paths)

        return buffer


    @classmethod
    def load_all_environments(cls, base_path: str = "data/mt50", buffer_size: int = 1000, video_shape: tuple = (480, 480), fps: float = 30.0, split: str = "train", load_env_name: str = None) -> Dict[str, 'ReplayBufferPKL']:
        env_buffers = {'info': {}}
        total_start_time = time.time()
        env_buffers['info']['num_envs'] = 0
        env_buffers['info']['num_tasks'] = 0

        pbar = tqdm(os.listdir(base_path), desc=f"Loading data for ={split}= environment", leave=True, dynamic_ncols=True)
        for item in pbar:
            if load_env_name is not None:
                if item != load_env_name:
                    continue
            item_path = os.path.join(base_path, item)
            pbar.set_postfix(environment=item)
            if os.path.isdir(item_path):
                env_name = item
                buffer = cls.load_from_path(
                    item_path, buffer_size, video_shape, fps, split
                )
                env_buffers[env_name] = buffer
                env_buffers['info']['num_envs'] += 1
                env_buffers['info']['num_tasks'] += buffer.num_tasks
                if env_buffers['info']['num_tasks'] > 50:
                    break
            pbar.update(1)

        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print(f"Total time to load all ={split}= environments: {total_time:.2f} seconds")
        return env_buffers


# Usage example
if __name__ == "__main__":
    base_path = "data/mt50"

    all_env_buffers = ReplayBufferPKL.load_all_environments(base_path=base_path)

    for env_name, buffer in all_env_buffers.items():
        print(f"Environment: {env_name}, Number of tasks: {len(buffer)}")