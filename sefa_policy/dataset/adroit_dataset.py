from typing import Dict
import torch
import numpy as np
import copy
import os
import pickle
import random
from sefa_policy.common.pytorch_util import dict_apply
from sefa_policy.common.replay_buffer import ReplayBuffer
from sefa_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from sefa_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from sefa_policy.dataset.base_dataset import BaseImageDataset

class AdroitDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            mode=None,
            sefa_name=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'img'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.mode = mode
        self.sefa_name = sefa_name
        if mode == 'sefa':
            print(f"AdroitDataset: Loading SeFA data from {sefa_name}")
            with open(f"data/adroit/{task_name}/{sefa_name}.pkl", "rb") as f:
                self.sefa_data = pickle.load(f)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
            'img': self.replay_buffer['img'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)
        img = sample['img'][:,].astype(np.float32)
        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
                'img': img.transpose(0, 3, 1, 2),
            },
            'action': sample['action'].astype(np.float32)
        }

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
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

