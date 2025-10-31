import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
# from sefa_policy.gym_util.async_vector_env import AsyncVectorEnv
# from sefa_policy.gym_util.sync_vector_env import SyncVectorEnv
from sefa_policy.gym_util.robomimic_multistep_wrapper import RobomimicMultiStepWrapper
from sefa_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from sefa_policy.model.common.rotation_transformer import RotationTransformer

from sefa_policy.policy.base_image_policy import BaseImagePolicy
from sefa_policy.common.pytorch_util import dict_apply
from sefa_policy.env_runner.base_image_runner import BaseImageRunner
from sefa_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env


class RobomimicImageLoopRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None or True:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return RobomimicMultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return RobomimicMultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        # env_init_fn_dills = list()
        env_video_paths = list()
        env_init_states = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis

                init_state = f[f'data/demo_{train_idx}/states'][0]
                # switch to init_state reset
                env_init_states.append(init_state)

                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env_video_paths.append(filename)
                else:
                    env_video_paths.append(None)

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            if enable_render:
                filename = pathlib.Path(output_dir).joinpath(
                    'media', wv.util.generate_id() + ".mp4")
                filename.parent.mkdir(parents=False, exist_ok=True)
                filename = str(filename)
                env_video_paths.append(filename)
            else:
                env_video_paths.append(None)

            # switch to seed reset
            env_init_states.append(None)
            env_seeds.append(seed)
            env_prefixs.append('test/')

        self.env_meta = env_meta
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_video_paths = env_video_paths
        self.env_init_states = env_init_states
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype

        # plan for rollout
        n_envs = len(self.env_fns)
        env_name = self.env_meta['env_name']

        # allocate data
        all_video_paths = []
        all_rewards = []
        all_success_rates = []  # New list to store success rates

        pbar = tqdm.tqdm(total=self.max_steps*n_envs, leave=False, mininterval=self.tqdm_interval_sec, dynamic_ncols=True)
        for task_idx in range(n_envs):
            current_env = self.env_fns[task_idx]()

            assert isinstance(current_env.env.env, RobomimicImageWrapper)
            if self.env_init_states[task_idx] is not None:
                current_env.env.env.init_state = self.env_init_states[task_idx]
            else:
                current_env.env.env.init_state = None
                current_env.seed(self.env_seeds[task_idx])

            obs = current_env.reset()
            image = current_env.env.env.render()
            rendered_imgs = [image]
            image = image / 255.0

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

            pbar.set_description(f"Eval RobomimicImageRunner | env {env_name}, task {task_idx} | success_rate: {success_count}/{n_envs}")
            done = False
            cnt = 0
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']    ## [1, 8, 10]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info, multistep_imgs = current_env.step(env_action[0])
                rendered_imgs += multistep_imgs
                past_action = action

                # if len(rendered_imgs) > 8:
                #     break
                # update pbar
                if done:
                    pbar.update(self.max_steps - action.shape[1] * cnt)
                else:
                    pbar.update(action.shape[1])
                    cnt += 1
            pbar.close()

            all_video_paths.append(self.env_video_paths[task_idx])
            # save rendered_imgs as mp4
            if self.env_video_paths[task_idx] is not None:
                if not video_recoder.is_ready():
                    video_recoder.start(self.env_video_paths[task_idx])
                # start = time.time()
                for img in rendered_imgs:
                    assert img.dtype == np.uint8
                    video_recoder.write_frame(img)
                # print('time to write frames: ', time.time()-start)
                if video_recoder.is_ready():
                    video_recoder.stop()
            # current_env.env.env.env.mujoco_renderer.close()

            all_rewards.append(reward)
            all_success_rates.append(done)
            pbar.set_description(f"Eval RobomimicImageRunner | env {env_name}, task {task_idx} | success_rate: {np.sum(all_success_rates)}/{n_envs}")
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
        for task_idx in range(n_envs):
            seed = self.env_seeds[task_idx]
            prefix = self.env_prefixs[task_idx]
            max_reward = np.max(all_rewards[task_idx])
            success_rate = all_success_rates[task_idx]
            max_rewards[prefix].append(max_reward)
            success_rates[prefix].append(success_rate)
            log_data[prefix+env_name+f'_sim_max_reward_{seed}'] = max_reward
            log_data[prefix+env_name+f'_sim_success_rate_{seed}'] = success_rate * 1.0  # Log individual success rates

            # visualize sim
            video_path = self.env_video_paths[task_idx]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+env_name+f'_sim_video_{seed}'] = sim_video

        
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

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
