from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from sefa_policy.policy.base_image_policy import BaseImagePolicy
from sefa_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from sefa_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from sefa_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import sefa_policy.model.vision.crop_randomizer as dmvc
from sefa_policy.common.pytorch_util import dict_apply, replace_submodules
from sefa_policy.dataset.mt10_sampler import Normalizer

from scipy import integrate
from sefa_policy.common import sde_lib

class SeFAUnetHybridImageMetaworldPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            rf_config,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # parameters passed to step
            mode='train',
            reflow_t_schedule='t0',
            sample_N=1,
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        self.agent_pos_shape = int(obs_shape_meta['agent_pos']['shape'][0])
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
        
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
        # print("Global cond dim: %d" % global_cond_dim)

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mode = mode
        self.reflow_t_schedule = reflow_t_schedule
        self.sample_N = sample_N
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = Normalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
        # Setup SDEs
        self.sde = sde_lib.RectifiedFlow(init_type=rf_config.sampling.init_type, noise_scale=rf_config.sampling.init_noise_scale, use_ode_sampler=rf_config.sampling.use_ode_sampler)
        self.sampling_eps = 1e-3



    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None, trajectory=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        with torch.no_grad():
            model = self.model
            rtol = atol = self.sde.ode_tol
            method = 'RK45'
            eps = 1e-3 if self.sampling_eps is None else self.sampling_eps

            if trajectory is None:
                z0 = (torch.randn(condition_data.shape)*self.sde.noise_scale).to(condition_data.device)
                trajectory = z0.detach().clone()

            trajectory[condition_mask] = condition_data[condition_mask]

            if self.mode == 'sefa':
                x = trajectory
                dt = 1. / self.sample_N

                for i in range(self.sample_N):
                    num_t = i / self.sample_N * (self.sde.T - eps) + eps
                    vec_t = torch.ones(condition_data.shape[0], device=x.device) * num_t
                    drift = model(x, vec_t*999, 
                        local_cond=local_cond, global_cond=global_cond)

                    # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability 
                    sigma_t = self.sde.sigma_t(num_t)
                    pred_sigma = drift + (sigma_t**2)/(2*(self.sde.noise_scale ** 2) * ((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * drift - 0.5 * (2.-num_t)*x.detach().clone())

                    x = x.detach().clone() + pred_sigma * dt + sigma_t * math.sqrt(dt) * torch.randn_like(pred_sigma).to(x.device)
                trajectory = x
            else:
                def ode_func(t, x):
                    x = torch.from_numpy(x.reshape(condition_data.shape)).to(condition_data.device).type(torch.float32)
                    vec_t = torch.ones(condition_data.shape[0], device=x.device) * t
                    drift = model(x, vec_t*999, 
                        local_cond=local_cond, global_cond=global_cond)

                    return drift.detach().cpu().numpy().reshape((-1,))

                if self.mode == 'gen_data':
                    noise = trajectory

                # Black-box ODE solver for the probability flow ODE
                solution = integrate.solve_ivp(ode_func, (eps, self.sde.T), trajectory.detach().cpu().numpy().reshape((-1,)),
                                                rtol=rtol, atol=atol, method=method)
                trajectory = torch.tensor(solution.y[:, -1]).reshape(condition_data.shape).to(condition_data.device).type(torch.float32)

            # finally make sure conditioning is enforced
            trajectory[condition_mask] = condition_data[condition_mask]

            if self.mode == 'gen_data':
                return trajectory, noise
            else:
                return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], action_embedding=None, object_embedding=None, gen_data=False) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nbaseobs = self.normalizer.normalize_obs(obs_dict["agent_pos"])
        nbaseobs = nbaseobs[:, :, :self.agent_pos_shape]
        nimages = self.normalizer.normalize_image(obs_dict["image"])
        nobs = {
            "agent_pos": nbaseobs,
            "image": nimages,
        }
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        if gen_data:
            nsample, noise = self.conditional_sample(
                cond_data,
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                **self.kwargs)
        else:
            nsample = self.conditional_sample(
                cond_data,
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer.unnormalize_action(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if gen_data:
            return noise, result['action_pred'], nobs_features.clone().detach().reshape(B, To, -1)
        else:
            return result

    # ========= training  ============
    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):

        # normalize input
        # assert 'valid_mask' not in batch
        nbaseobs = self.normalizer.normalize_obs(batch["base_obs"])
        nbaseobs = nbaseobs[:, :, :self.agent_pos_shape]
        nactions = self.normalizer.normalize_action(batch['actions'])
        nimages = self.normalizer.normalize_image(batch['imgs'])
        nobs = {
            "agent_pos": nbaseobs,
            "image": nimages,
        }
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.mode == 'sefa' and self.training:
            pass
        elif self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].float().reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        if self.mode == 'sefa' and self.training:
            noise = batch['z0']
            trajectory = batch['z1'].detach().clone()
            global_cond = batch['cond'].reshape(batch_size, -1).detach().clone()
            eps = 1e-3 if self.sampling_eps is None else self.sampling_eps
            if self.reflow_t_schedule=='t0':
                t = torch.zeros(trajectory.shape[0], device=trajectory.device) * (self.sde.T - eps) + eps
            else:
                assert False, 'Not implemented'
        else:
            noise = torch.randn(trajectory.shape, device=trajectory.device) * self.sde.noise_scale
            ### standard rectified flow loss
            t = torch.rand(trajectory.shape[0], device=trajectory.device) * (self.sde.T - self.sampling_eps) + self.sampling_eps
        t_expand = t.view(-1, 1, 1).repeat(1, trajectory.shape[1], trajectory.shape[2])
        perturbed_data = t_expand * trajectory + (1. - t_expand) * noise
        target = trajectory - noise

        noisy_trajectory = perturbed_data
        timesteps = t * 999

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'rf':
            pass
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        losses = F.mse_loss(pred, target, reduction='none')
        loss = losses * loss_mask.type(losses.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
