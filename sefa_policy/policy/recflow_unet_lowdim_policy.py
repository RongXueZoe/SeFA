from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from sefa_policy.model.common.normalizer import LinearNormalizer
from sefa_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from sefa_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from sefa_policy.model.diffusion.mask_generator import LowdimMaskGenerator

from scipy import integrate
from sefa_policy.common import sde_lib

class SeFAUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            rf_config,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            mode='train',
            reflow_t_schedule='t0',
            sample_N=1,
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.mode = mode
        self.reflow_t_schedule = reflow_t_schedule
        self.sample_N = sample_N
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

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
                # nfe = solution.nfev
                trajectory = torch.tensor(solution.y[:, -1]).reshape(condition_data.shape).to(condition_data.device).type(torch.float32)

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        if self.mode == 'gen_data':
            return trajectory, noise
        else:
            return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], gen_data=False) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
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
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        if gen_data:
            return noise, result['action_pred'], nobs.clone().detach().reshape(B, To, -1)
        else:
            return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']
        batch_size = action.shape[0]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.mode == 'sefa' and self.training:
            pass
        elif self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        if self.mode == 'sefa' and self.training:
            noise = batch['z0']
            trajectory = batch['z1'].detach().clone()
            global_cond = batch['cond'].reshape(batch_size, -1).detach().clone()
            # batch = data.detach().clone()
            eps = 1e-3 if self.sampling_eps is None else self.sampling_eps
            if self.reflow_t_schedule=='t0': ### distill for t = 0 (k=1)
                t = torch.zeros(trajectory.shape[0], device=trajectory.device) * (self.sde.T - eps) + eps
            # elif self.reflow_t_schedule=='t1': ### reverse distill for t=1 (fast embedding)
            #     t = torch.ones(batch.shape[0], device=batch.device) * (self.sde.T - eps) + eps
            # elif self.reflow_t_schedule=='uniform': ### train new rectified flow with reflow
            #     t = torch.rand(batch.shape[0], device=batch.device) * (self.sde.T - eps) + eps
            # elif type(self.reflow_t_schedule)==int: ### k > 1 distillation
            #     t = torch.randint(0, self.reflow_t_schedule, (batch.shape[0], ), device=batch.device) * (self.sde.T - eps) / self.reflow_t_schedule + eps
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
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
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

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
