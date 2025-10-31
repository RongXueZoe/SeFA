import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import metaworld
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from metaworld.policies.sawyer_push_v2_policy import  SawyerPushV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.policies.sawyer_peg_insertion_side_v2_policy import SawyerPegInsertionSideV2Policy
from metaworld.policies.sawyer_window_open_v2_policy import SawyerWindowOpenV2Policy
from metaworld.policies.sawyer_window_close_v2_policy import SawyerWindowCloseV2Policy
from metaworld.policies.sawyer_door_open_v2_policy import SawyerDoorOpenV2Policy
from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy
from metaworld.policies.sawyer_drawer_close_v2_policy import SawyerDrawerCloseV2Policy
from metaworld.policies.sawyer_button_press_topdown_v2_policy import SawyerButtonPressTopdownV2Policy
import numpy as np
import torch.nn as nn
import torch
from .ray_sampler import RaySampler


policies = {
    "reach-v2": SawyerReachV2Policy(),
    "push-v2": SawyerPushV2Policy(),
    "pick-place-v2": SawyerPickPlaceV2Policy(),
    "door-open-v2": SawyerDoorOpenV2Policy(),
    "drawer-open-v2": SawyerDrawerOpenV2Policy(),
    "drawer-close-v2": SawyerDrawerCloseV2Policy(),
    "button-press-topdown-v2": SawyerButtonPressTopdownV2Policy(),
    "peg-insert-side-v2": SawyerPegInsertionSideV2Policy(),
    "window-open-v2": SawyerWindowOpenV2Policy(),
    "window-close-v2": SawyerWindowCloseV2Policy(),
}

mt10 = metaworld.MT10()
mt_custom_train_envs = mt10.train_classes

train_env_list = []
policy_list = []
name_list = []
max_path_length = 0
for env_idx, name in enumerate(mt_custom_train_envs):
    env_cls = mt_custom_train_envs[name]
    env = env_cls()
    policy_list.append(policies[name])
    name_list.append(name)

    all_tasks = [task for task in mt10.train_tasks if task.env_name == name]
    env.render_mode = "rgb_array"
    env.all_tasks = all_tasks[:25]
    env.mujoco_renderer.camera_id = 4
    max_path_length = max(max_path_length, env.max_path_length)
    train_env_list.append(env)


train_sampler = RaySampler(
    agents=policy_list,
    envs=train_env_list,
    names=name_list,
    batch_size=32 // len(train_env_list),
    win_len=16,
    max_episode_length=max_path_length,
    n_workers=len(train_env_list),
)

mt_custom_test_envs = mt10.train_classes
test_env_list = []
test_policy_list = []
test_name_list = []
max_path_length = 0
for env_idx, name in enumerate(mt_custom_test_envs):
    env_cls = mt_custom_test_envs[name]
    env = env_cls()
    env.render_mode = "rgb_array"
    all_tasks = [task for task in mt10.test_tasks if task.env_name == name]
    env.all_tasks = all_tasks[25:]
    env.mujoco_renderer.camera_id = 4
    test_policy_list.append(policies[name])
    test_name_list.append(name)
    max_path_length = max(max_path_length, env.max_path_length)

    test_env_list.append(env)

test_sampler = RaySampler(
    agents=test_policy_list,
    envs=test_env_list,
    names=test_name_list,
    batch_size=32 // len(test_env_list),
    win_len=16,
    max_episode_length=max_path_length,
    n_workers=len(test_env_list),
)


obs_low = train_env_list[0].observation_space.low
obs_high = train_env_list[0].observation_space.high
obs_low = np.where(np.isneginf(obs_low), 0.0, obs_low)
obs_high = np.where(np.isinf(obs_high), 0.0, obs_high)


obs_mean = (obs_low + obs_high) / 2
obs_std = (obs_high - obs_mean)
obs_std[obs_std == 0] = 1

act_low = train_env_list[0].action_space.low
act_high = train_env_list[0].action_space.high
act_low = np.where(np.isneginf(act_low), 0.0, act_low)
act_high = np.where(np.isinf(act_high), 0.0, act_high)
act_mean = (act_low + act_high) / 2
act_std = (act_high - act_mean)
act_std[act_std == 0] = 1

img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])

class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("obs_mean", torch.from_numpy(obs_mean).float())
        self.register_buffer("obs_std", torch.from_numpy(obs_std).float())
        
        self.register_buffer("act_mean", torch.from_numpy(act_mean).float())
        self.register_buffer("act_std", torch.from_numpy(act_std).float())

        self.register_buffer("img_mean", torch.from_numpy(img_mean).float())
        self.register_buffer("img_std", torch.from_numpy(img_std).float())  

    def normalize_obs(self, x):
        return (x - self.obs_mean) / self.obs_std
    
    def normalize_action(self, x):
        return (x - self.act_mean) / self.act_std

    def normalize_image(self, x):
        return (x - self.img_mean[None, None, :, None, None]) / self.img_std[None, None, :, None, None]

    def unnormalize_action(self, x):
        return x * self.act_std + self.act_mean
