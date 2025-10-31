import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from sefa_policy.gym_util.metaworld_multistep_wrapper import MetaworldMultiStepWrapper
from sefa_policy.gym_util.gymnasium_video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from sefa_policy.policy.base_image_policy import BaseImagePolicy
from sefa_policy.common.pytorch_util import dict_apply
from sefa_policy.env_runner.base_image_runner import BaseImageRunner

import pickle
from PIL import Image
import types
import time
import mujoco
from copy import deepcopy
import glob

import metaworld
from metaworld.policies import *  # noqa: F403
mt50 = metaworld.MT50()
mt_custom_test_envs = mt50.train_classes

policies = {
    "assembly-v2": SawyerAssemblyV2Policy(),
    "basketball-v2": SawyerBasketballV2Policy(),
    "bin-picking-v2": SawyerBinPickingV2Policy(),
    "box-close-v2": SawyerBoxCloseV2Policy(),
    "button-press-topdown-v2": SawyerButtonPressTopdownV2Policy(),
    "button-press-topdown-wall-v2": SawyerButtonPressTopdownWallV2Policy(),
    "button-press-v2": SawyerButtonPressV2Policy(),
    "button-press-wall-v2": SawyerButtonPressWallV2Policy(),
    "coffee-button-v2": SawyerCoffeeButtonV2Policy(),
    "coffee-pull-v2": SawyerCoffeePullV2Policy(),
    "coffee-push-v2": SawyerCoffeePushV2Policy(),
    "dial-turn-v2": SawyerDialTurnV2Policy(),
    "disassemble-v2": SawyerDisassembleV2Policy(),
    "door-close-v2": SawyerDoorCloseV2Policy(),
    "door-lock-v2": SawyerDoorLockV2Policy(),
    "door-open-v2": SawyerDoorOpenV2Policy(),
    "door-unlock-v2": SawyerDoorUnlockV2Policy(),
    "hand-insert-v2": SawyerHandInsertV2Policy(),
    "drawer-close-v2": SawyerDrawerCloseV2Policy(),
    "drawer-open-v2": SawyerDrawerOpenV2Policy(),
    "faucet-open-v2": SawyerFaucetOpenV2Policy(),
    "faucet-close-v2": SawyerFaucetCloseV2Policy(),
    "hammer-v2": SawyerHammerV2Policy(),
    "handle-press-side-v2": SawyerHandlePressSideV2Policy(),
    "handle-press-v2": SawyerHandlePressV2Policy(),
    "handle-pull-side-v2": SawyerHandlePullSideV2Policy(),
    "handle-pull-v2": SawyerHandlePullV2Policy(),
    "lever-pull-v2": SawyerLeverPullV2Policy(),
    "pick-place-wall-v2": SawyerPickPlaceWallV2Policy(),
    "pick-out-of-hole-v2": SawyerPickOutOfHoleV2Policy(),
    "pick-place-v2": SawyerPickPlaceV2Policy(),
    "plate-slide-v2": SawyerPlateSlideV2Policy(),
    "plate-slide-side-v2": SawyerPlateSlideSideV2Policy(),
    "plate-slide-back-v2": SawyerPlateSlideBackV2Policy(),
    "plate-slide-back-side-v2": SawyerPlateSlideBackSideV2Policy(),
    "peg-insert-side-v2": SawyerPegInsertionSideV2Policy(),
    "peg-unplug-side-v2": SawyerPegUnplugSideV2Policy(),
    "soccer-v2": SawyerSoccerV2Policy(),
    "stick-push-v2": SawyerStickPushV2Policy(),
    "stick-pull-v2": SawyerStickPullV2Policy(),
    "push-v2": SawyerPushV2Policy(),
    "push-wall-v2": SawyerPushWallV2Policy(),
    "push-back-v2": SawyerPushBackV2Policy(),
    "reach-v2": SawyerReachV2Policy(),
    "reach-wall-v2": SawyerReachWallV2Policy(),
    "shelf-place-v2": SawyerShelfPlaceV2Policy(),
    "sweep-into-v2": SawyerSweepIntoV2Policy(),
    "sweep-v2": SawyerSweepV2Policy(),
    "window-open-v2": SawyerWindowOpenV2Policy(),
    "window-close-v2": SawyerWindowCloseV2Policy(),
}

def resize_img(img):
    im_pil = Image.fromarray(img)
    im_pil = im_pil.resize((96, 96), Image.ANTIALIAS)
    img = np.array(im_pil)
    return np.transpose(img, (2, 0, 1))


def step_img(self, action):
    agent_pos, reward, terminated, truncated, info = self.step(action)
    img = self.render()
    img = resize_img(img) / 255.0
    obs = {"agent_pos": agent_pos, "image": img}
    return obs, reward, terminated, truncated, info


def reset_img(self):
    agent_pos, info = self.reset()
    img = self.render()
    img = resize_img(img) / 255.0
    obs = {"agent_pos": agent_pos, "image": img}
    return obs, info


class MT50ImageLoopRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            env_name=None,
            camera_id=1,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            num_env=10,
        ):
        super().__init__(output_dir)
        tasks_f = glob.glob(f"./data/mt50/{env_name}/task-*.pkl")
        n_train = min(len(tasks_f), 5)
        num_task = n_train + n_test
        n_envs = num_env * num_task
        self.camera_id = camera_id

        steps_per_render = max(10 // fps, 1)
        def get_env_cls(task):
            def env_fn():
                env_cls = mt_custom_test_envs[env_name]
                env = env_cls()
                env.render_mode = "rgb_array"
                env.max_path_length = max_steps
                all_tasks = mt50.train_tasks
                env.all_tasks = all_tasks

                env.set_task(task)

                return MetaworldMultiStepWrapper(
                    VideoRecordingWrapper(
                        env,
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec="h264",
                            input_pix_fmt="rgb24",
                            crf=crf,
                            thread_type="FRAME",
                            thread_count=1,
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render,
                    ),
                    n_obs_steps=n_obs_steps,
                    n_action_steps=n_action_steps,
                    max_episode_steps=max_steps,
                )
            return env_fn

        env_fns = []
        all_tasks = [task for task in mt50.train_tasks if task.env_name == env_name]
        for task_idx in range(n_test):
            task = all_tasks[task_idx + 25]
            env_fns.append(get_env_cls(task))
        for task_f in tasks_f[:n_train]:
            task = pickle.load(open(task_f, "rb"))
            env_fns.append(get_env_cls(task))
        env_prefixs = list()
        env_video_paths = []

        # train
        for i in range(n_train):
            enable_render = i < n_train_vis

            if enable_render:
                filename = pathlib.Path(output_dir).joinpath(
                    'media', env_name.split('-v2')[0] + '_train_' + wv.util.generate_id() + ".mp4")
                filename.parent.mkdir(parents=False, exist_ok=True)
                filename = str(filename)
                env_video_paths.append(filename)
            else:
                env_video_paths.append(None)

            env_prefixs.append('train/')

        # test
        for i in range(n_test):
            enable_render = i < n_test_vis

            if enable_render:
                filename = pathlib.Path(output_dir).joinpath(
                    'media', env_name.split('-v2')[0] + '_test_' + wv.util.generate_id() + ".mp4")
                filename.parent.mkdir(parents=False, exist_ok=True)
                filename = str(filename)
                env_video_paths.append(filename)
            else:
                env_video_paths.append(None)
            
            env_prefixs.append('test/')

        self.env_fns = env_fns
        self.env_name = env_name
        self.env_video_paths = env_video_paths
        self.env_prefixs = env_prefixs
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseImagePolicy, expert_policy=None):
        device = policy.device
        dtype = policy.dtype

        # plan for rollout
        n_envs = len(self.env_fns)

        # allocate data
        all_video_paths = []
        all_rewards = []
        all_success_rates = []  # New list to store success rates

        pbar = tqdm.tqdm(total=self.max_steps*n_envs, leave=False, mininterval=self.tqdm_interval_sec, dynamic_ncols=True)
        for i in range(n_envs):
            current_env = self.env_fns[i]()
            env_name = self.env_name
            expert_env = mt50.train_classes[env_name]()
            obs, _, _env = current_env.reset()
            current_env.env.env.render_mode = "rgb_array"
            current_env.env.env.mujoco_renderer.camera_id = self.camera_id
            image = current_env.env.env.render()
            rendered_imgs = [image]
            image = resize_img(image) / 255.0

            past_action = None
            policy.reset()
            success_rate = 0

            video_recoder=VideoRecorder.create_h264(
                fps=self.fps,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=self.crf,
                thread_type="FRAME",
                thread_count=1,
            )

            if len(all_success_rates) == 0:
                success_count = 0
            else:
                success_count = np.sum(all_success_rates)
            pbar.set_description(f"Eval MT50ImageRunner | env {env_name}, task {i} | success_rate: {success_count}/{n_envs}")
            done = False
            cnt = 0
            while not done:

                ## initialize expert env
                state = current_env.env.env.__getstate__()
                expert_env._freeze_rand_vec = False
                expert_env._set_task_called = True
                expert_env._set_task_inner()
                expert_env.__setstate__(state)
                expert_env.mujoco_renderer.camera_id = current_env.env.env.mujoco_renderer.camera_id
                expert_env.render_mode = current_env.env.env.render_mode
                expert_env._set_task_called = True
                expert_env._freeze_rand_vec = True
                expert_env._freeze_rand_vec = current_env.env.env._freeze_rand_vec
                expert_env._last_rand_vec = current_env.env.env._last_rand_vec
                expert_env._partially_observable = current_env.env.env._partially_observable
                try:
                    expert_env.model.site('goal').pos = current_env.env.env.model.site('goal').pos
                except:
                    pass
                expert_env.model = deepcopy(current_env.env.env.model)
                expert_env.data = deepcopy(current_env.env.env.data)
                mujoco.mj_forward(expert_env.model, expert_env.data)

                assert np.allclose(expert_env.init_qpos, current_env.env.env.init_qpos, atol=1e-6, rtol=1e-5)
                assert np.allclose(expert_env.init_qvel, current_env.env.env.init_qvel, atol=1e-6, rtol=1e-5)
                assert np.allclose(expert_env.get_env_state()[0], current_env.env.env.get_env_state()[0], atol=1e-10, rtol=1e-5)
                assert np.allclose(expert_env.get_env_state()[1], current_env.env.env.get_env_state()[1], atol=1e-10, rtol=1e-5)

                # create obs dict
                np_obs_dict = dict(agent_pos=obs, image=image)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)

                # device transfer
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x.copy()).float().to(device=device))

                if obs_dict['image'].ndim == 3:
                    obs_dict['image'] = torch.stack([obs_dict['image'].unsqueeze(0), obs_dict['image'].unsqueeze(0)], axis=1)
                elif obs_dict['image'].ndim == 4:
                    obs_dict['image'] = obs_dict['image'].unsqueeze(0)
                if obs_dict['agent_pos'].ndim == 2:
                    obs_dict['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                assert obs_dict['image'].ndim == 5, '`image` shape: {}'.format(obs_dict['image'].shape)
                assert obs_dict['agent_pos'].ndim == 3, '`agent_pos` shape: {}'.format(obs_dict['agent_pos'].shape)

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, terminated, truncated, success, info, _envs, multistep_imgs = current_env.step(action[0])
                done = np.all(terminated + truncated + success)
                past_action = action
                rendered_imgs += multistep_imgs
                if len(multistep_imgs) > 1:
                    second_last_img = resize_img(multistep_imgs[-2]) / 255.0
                else:
                    second_last_img = resize_img(multistep_imgs[-1]) / 255.0
                last_img = resize_img(multistep_imgs[-1]) / 255.0
                image = np.stack([second_last_img, last_img], axis=0)    ## [2, 3, 96, 96]

                # update pbar
                if done:
                    pbar.update(self.max_steps - action.shape[1] * cnt)
                else:
                    pbar.update(action.shape[1])
                    cnt += 1

            all_video_paths.append(self.env_video_paths[i])
            # save rendered_imgs as mp4
            if self.env_video_paths[i] is not None:
                if not video_recoder.is_ready():
                    video_recoder.start(self.env_video_paths[i])
                for img in rendered_imgs:
                    assert img.dtype == np.uint8
                    video_recoder.write_frame(img)
                if video_recoder.is_ready():
                    video_recoder.stop()

            all_rewards.append(reward)
            all_success_rates.append(success)
            pbar.set_description(f"Eval MT50ImageRunner | env {env_name}, task {i} | success_rate: {np.sum(all_success_rates)}/{n_envs}")
        pbar.close()

        # log
        max_rewards = collections.defaultdict(list)
        success_rates = collections.defaultdict(list)  # New dictionary for success rates
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_envs):
            prefix = self.env_prefixs[i]
            env_name = self.env_name
            env_name = env_name.split('-v2')[0]
            max_reward = np.max(all_rewards[i])
            success_rate = all_success_rates[i]
            max_rewards[prefix].append(max_reward)
            success_rates[prefix].append(success_rate)
            log_data[prefix+env_name+f'_sim_max_reward_{i}'] = max_reward
            log_data[prefix+env_name+f'_sim_success_rate_{i}'] = success_rate * 1.0  # Log individual success rates

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+env_name+f'_sim_video_{i}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        for prefix, value in success_rates.items():
            name = prefix + 'mean_success_rate'
            value = np.mean(value)
            log_data[name] = value  # Log mean success rate

        return log_data
