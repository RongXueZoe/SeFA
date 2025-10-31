from typing import Dict
import torch
import numpy as np
import copy
import os
import pickle
import random
from sefa_policy.common.pytorch_util import dict_apply
from sefa_policy.common.replay_buffer_pkl import ReplayBufferPKL
from sefa_policy.dataset.base_dataset import BaseImageDataset

class MT50ImageDataset(BaseImageDataset):
    def __init__(self, 
            base_path,
            env_name=None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            buffer_size: int=1000,
            video_shape: tuple=(480, 480),
            fps: float=30.0,
            seed=42,
            val_ratio=0.0,
            split='train',
            repeat_times=1,
            mode='train',
            sefa_name=None,
            **kwargs,
            ):
        super().__init__()
        self.replay_buffer = ReplayBufferPKL.load_all_environments(
            base_path=base_path, buffer_size=buffer_size, video_shape=video_shape, fps=fps, split=split, load_env_name=env_name)

        if mode == 'sefa':
            print(f"Loading SeFA data from data/mt50/{env_name}/{sefa_name}.pkl")
            with open(f"data/mt50/{env_name}/{sefa_name}.pkl", "rb") as f:
                self.sefa_data = pickle.load(f)

        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.mode = mode
        self.repeat_times = repeat_times if split == 'train' else 1

    def __len__(self) -> int:
        return int(self.replay_buffer['info']['num_tasks'] * self.repeat_times)

    def _buffer_to_data(self, sample, demo_idx):
        data = sample.paths[demo_idx]
        all_imgs = sample.imgs[demo_idx]
        if isinstance(all_imgs, list):
            all_imgs = np.stack(all_imgs, axis=0)

        agent_pos = data['obs'].astype(np.float32)
        image = all_imgs / 255.0
        actions = data['actions']

        p = np.random.rand()
        if p < 0.1:
            # select last horizon frames and pad with zeros
            pad_len = np.random.randint(0, self.horizon - 3)
            image = np.pad(image, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
            agent_pos = np.pad(agent_pos, ((0, pad_len), (0, 0)))
            actions = np.pad(actions, ((0, pad_len), (0, 0)))
            image = image[-self.horizon :]
            agent_pos = agent_pos[-self.horizon:]
            actions = actions[-self.horizon:]
        else:
            # select random horizon frames
            frame_indices = np.arange(len(image) - self.horizon)
            select_start_idx = np.random.choice(frame_indices)
            image = image[select_start_idx:select_start_idx + self.horizon]
            agent_pos = agent_pos[select_start_idx:select_start_idx + self.horizon]
            actions = actions[select_start_idx:select_start_idx + self.horizon]

        assert len(image) == self.horizon, f"image length {len(image)} != horizon {self.horizon}"
        assert len(agent_pos) == self.horizon, f"agent_pos length {len(agent_pos)} != horizon {self.horizon}"
        assert len(actions) == self.horizon, f"actions length {len(actions)} != horizon {self.horizon}"

        data = {
            "imgs": image,
            "base_obs": agent_pos,
            "actions": actions.astype(np.float32),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        all_num_demos = int(self.replay_buffer['info']['num_tasks'])
        if idx >= all_num_demos:
            idx = idx % all_num_demos

        # Find the correct environment and demo index
        current_count = 0
        for env_name, buffer in self.replay_buffer.items():
            if env_name == 'info':
                continue
            if current_count + buffer.num_tasks > idx:
                # This is the correct environment
                demo_idx = idx - current_count
                break
            current_count += buffer.num_tasks
        else:
            raise IndexError(f"Index {idx} not found in any environment")

        data = self._buffer_to_data(buffer, demo_idx)

        torch_data = dict_apply(data, torch.from_numpy)

        torch_data['idx'] = idx
        if self.mode == 'sefa':
            z0_list = self.sefa_data['z0_cllt'][idx]
            z1_list = self.sefa_data['z1_cllt'][idx]
            cond_list = self.sefa_data['cond_cllt'][idx]
            rand_id = random.randint(0, len(z0_list) - 1)
            torch_data['z0'] = torch.from_numpy(z0_list[rand_id])
            torch_data['z1'] = torch.from_numpy(z1_list[rand_id])
            torch_data['cond'] = torch.from_numpy(cond_list[rand_id])
        return torch_data
