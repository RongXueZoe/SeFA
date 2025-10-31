import os
import numpy as np
import pickle
import torch
from torchvision.models import resnet18
from sefa_policy.common.pytorch_util import dict_apply
from sefa_policy.dataset.mt10_sampler import Normalizer
from tqdm import tqdm

normalizer = Normalizer()

def load_all_samples(path):
    action_sequences = []
    target_poses = []
    sampled_video_paths = []
    for filename in os.listdir(path):
        if filename.endswith('.pkl') and not filename.startswith('task') and 'seed' in filename:
            with open(os.path.join(path, filename), "rb") as f:
                path_data = pickle.load(f)
                sampled_action = path_data['actions']    ## [97, 4]
                video_path = path_data['video_path'].replace('.mp4', '_96.npz')
                sampled_video_paths.append(video_path)
                pad_len = 102 - sampled_action.shape[0]    ## TODO: 102
                target_pos = np.pad(path_data['obs'], ((0, pad_len), (0, 0)))
                sampled_action = np.pad(sampled_action, ((0, pad_len), (0, 0)))
                action_sequences.append(torch.from_numpy(sampled_action))
                target_poses.append(torch.from_numpy(target_pos))
    action_sequences = torch.stack(action_sequences)    ## [num_sequences, 102, 4]
    target_poses = torch.stack(target_poses)    ## [num_sequences, 102, 39]

    return action_sequences, target_poses, sampled_video_paths


def find_nearest_action_sequence(sefa_action, sefa_cond, path, obs_encoder, env_name, sefa_name, action_sequences, target_poses, sampled_video_paths):
    # Load all action sequences from the given path
    if sefa_action is None or sefa_cond is None:
        sefa_path = f"data/mt50/{env_name}/{sefa_name}.pkl"
        with open(sefa_path, "rb") as f:
            sefa_data = pickle.load(f)
            sefa_action = torch.from_numpy(sefa_data['z1_cllt'][0][0])
            sefa_cond = torch.from_numpy(sefa_data['cond_cllt'][0][0])

    if path is not None:
        action_sequences, target_poses, sampled_video_paths = load_all_samples(path)

    action_near_pair = []
    min_action_dist = 0.1    ## Threshold for action distance
    for step in tqdm(range(action_sequences.shape[1] - sefa_action.shape[0] + 1), dynamic_ncols=True, leave=False):
        action_dist_lst = torch.norm(sefa_action - action_sequences[:, step:step+sefa_action.shape[0], :], dim=2)    ## [num_sequences, 16]
        action_dist_lst = torch.mean(action_dist_lst, dim=1)    ## [num_sequences]
        if min(action_dist_lst) < min_action_dist:
            for i, action_dist in enumerate(action_dist_lst):
                if action_dist < min_action_dist:
                    action_near_pair.append((i, step, action_sequences[i, step:step+sefa_action.shape[0], :], action_dist))

    if len(action_near_pair) == 0:
        return None
    assert obs_encoder is not None
    obs_encoder.eval()
    pos_lst = []
    frames_lst = []
    for item in tqdm(action_near_pair, dynamic_ncols=True, leave=False):
        frames = np.load(sampled_video_paths[item[0]])['arr_0'][item[1]:item[1]+sefa_action.shape[0], ...] / 255.0
        pad_len = 16 - frames.shape[0]
        if pad_len > 0:
            frames = torch.from_numpy(np.pad(frames, ((0, pad_len), (0, 0), (0, 0), (0, 0)))).float()
        else:
            frames = torch.from_numpy(frames).float()

        pos = target_poses[item[0], item[1]:item[1]+sefa_action.shape[0]]
        pos = normalizer.normalize_obs(pos)[:, :4]
        pos_lst.append(pos)
        frames = normalizer.normalize_image(frames)[0]
        frames_lst.append(frames)
    nobs = {
        "agent_pos": torch.stack(pos_lst),
        "image": torch.stack(frames_lst),
    }
    this_nobs = dict_apply(nobs, lambda x: x[:,:2,...].reshape(-1,*x.shape[2:]).cuda())
    with torch.no_grad():
        nobs_features = obs_encoder(this_nobs)
    dist = torch.norm(sefa_cond.cuda().reshape(-1) - nobs_features.reshape(-1, 2 * 68), dim=1)    ## [2*num_alike]
    min_feature_dist = 1e5
    if min(dist) < 1e-3:
        action_dist = 0
        for i, d in enumerate(dist):
            if d < 1e-3:
                if d < min_feature_dist:
                    min_feature_dist = d.item()
                    action_dist = action_near_pair[torch.argmin(dist).item()][3]
                    print(d.item(), action_dist.item())
                    nearest_sequence = action_near_pair[torch.argmin(dist).item()][2]
        if action_dist == 0:
            return None
        # Return the nearest action sequence
        return nearest_sequence
    else:
        return None


if __name__ == '__main__':
    action = np.array([0.1, 0.2, 0.3, 0.6])
    path = "data/mt50/assembly-v2"
    nearest_sequence = find_nearest_action_sequence(action, path)
    print(nearest_sequence)