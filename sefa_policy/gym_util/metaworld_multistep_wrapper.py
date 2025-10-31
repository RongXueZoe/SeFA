import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import defaultdict, deque
import dill

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

def aggregate(data, method='max'):
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


def stack_last_n_imgs(all_imgs, n_steps):
    assert(len(all_imgs) > 0)
    all_imgs = list(all_imgs)
    result = np.zeros((n_steps,) + all_imgs[-1].shape, 
        dtype=all_imgs[-1].dtype)
    start_idx = -min(n_steps, len(all_imgs))
    result[start_idx:] = np.array(all_imgs[start_idx:])
    if n_steps > len(all_imgs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MetaworldMultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.success = list()
        self.terminated = list()
        self.truncated = list()
        self.imgs = deque(maxlen=n_obs_steps+1)
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs, info, _env = self.env.reset()
        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.success = list()
        self.terminated = list()
        self.truncated = list()
        # self.imgs = deque([img], maxlen=self.n_obs_steps+1)
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        obs = self._get_obs(self.n_obs_steps)
        # img = self._get_imgs(self.n_obs_steps)
        return obs, None, _env

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        imgs = []
        for act in action:
            if len(self.success) > 0 and self.success[-1]:
                # termination
                break
            elif len(self.truncated) > 0 and self.truncated[-1]:
                # truncation
                break
            elif len(self.terminated) > 0 and self.terminated[-1]:
                # termination
                break
            observation, reward, terminated, truncated, info, _env = super().step(act)
            success = info['success'] > 0.5
            self._env = _env
            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                truncated = True
            self.terminated.append(terminated)
            self.truncated.append(truncated)
            self.success.append(success)
            self._add_info(info)
            imgs.append(self.env.env.render())

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        terminated = aggregate(self.terminated, 'max')
        truncated = aggregate(self.truncated, 'max')
        success = aggregate(self.success, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, terminated, truncated, success, info, self._env, imgs

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _get_imgs(self, n_steps=1):
        assert(len(self.imgs) > 0)
        return stack_last_n_imgs(self.imgs, n_steps)

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result

    def seed(self, seed=None):
        return self.env.seed(seed)
