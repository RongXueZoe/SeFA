if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import pickle

from sefa_policy.common.pytorch_util import dict_apply, optimizer_to
from sefa_policy.workspace.base_workspace import BaseWorkspace
from sefa_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from sefa_policy.dataset.base_dataset import BaseLowdimDataset
from sefa_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from sefa_policy.common.checkpoint_util import TopKCheckpointManager
from sefa_policy.common.json_logger import JsonLogger
from sefa_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

from scripts.find_nearest import find_nearest_action_sequence, load_all_samples

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainSeFAUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            cfg.use_wandb = False
            cfg.task.env_runner.n_envs = 1
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        if cfg.mode == 'gen_data' or cfg.mode == 'sefa':
            base_ckpt = cfg.base_ckpt
            assert base_ckpt is not None
            sefa_name = cfg.sefa_name
            assert sefa_name is not None
            self.load_checkpoint(path=base_ckpt)
            print(f"Loaded base checkpoint {base_ckpt}")
            if cfg.mode == 'sefa':
                self.epoch = 0
                self.global_step = 0
        else:
            assert cfg.mode in ['train', 'sefa'], f"Invalid mode: {cfg.mode}"

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        if cfg.use_wandb:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.mode == 'refine':
            print(f"Loading SeFA data from {cfg.sefa_name} to refine")
            rf2_path = f"data/kitchen/{cfg.task.env_name}/{cfg.sefa_name}.pkl"
            action_sequences, target_poses, sampled_video_paths = load_all_samples(f"data/kitchen/{cfg.task.env_name}")
            z0_cllt = {}
            z1_cllt = {}
            cond_cllt = {}
            with open(rf2_path, "rb") as f:
                rf2_data = pickle.load(f)
                z0_cllt = rf2_data['z0_cllt']
                z1_cllt = rf2_data['z1_cllt']
                cond_cllt = rf2_data['cond_cllt']
                print(len(z1_cllt), len(z1_cllt[0]))
                for outer_idx in tqdm.tqdm(range(len(z1_cllt)), dynamic_ncols=True, leave=True):
                    for inner_idx in tqdm.tqdm(range(len(z1_cllt[outer_idx])), dynamic_ncols=True, leave=False):
                        rf2_action = torch.from_numpy(z1_cllt[outer_idx][inner_idx])
                        rf2_cond = torch.from_numpy(cond_cllt[outer_idx][inner_idx])
                        nearest_sequence = find_nearest_action_sequence(rf2_action, rf2_cond, None, self.model.obs_encoder, cfg.task.env_name, cfg.sefa_name, action_sequences, target_poses, sampled_video_paths)
                        if nearest_sequence is not None:
                            z1_cllt[outer_idx][inner_idx] = nearest_sequence.cpu().numpy()
            gen_data = {"z0_cllt": z0_cllt, "z1_cllt": z1_cllt, "cond_cllt": cond_cllt}
            
            with open(f"data/kitchen/{cfg.task.env_name}/{cfg.sefa_name}_refined.pkl", "wb") as f:
                pickle.dump(gen_data, f)
            print(f"Generated RecFlow data saved to {f.name}")
            return

        # gen_data loop
        if cfg.mode == 'gen_data':
            self.model.eval()
            z0_cllt = {}
            z1_cllt = {}
            cond_cllt = {}
            if cfg.task.load_from_pkl:
                    with tqdm.tqdm(train_dataloader, desc=f"Generating SeFA data epoch={self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec, dynamic_ncols=True) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            # device transfer
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True).float())
                            agent_pos = batch["base_obs"]
                            if train_sampling_batch is None:
                                train_sampling_batch = batch

                            obs_dict = {
                                "agent_pos": agent_pos,
                                "image": batch["imgs"].float(),
                            }
                            for _ in range(cfg.sample_per_cond):
                                noise, action_pred, cond = self.model.predict_action(obs_dict, gen_data=True)

                                for bid, idx_th in enumerate(batch['idx']):
                                    idx = int(idx_th.item())
                                    if idx not in z0_cllt:
                                        z0_cllt[idx] = []
                                        z1_cllt[idx] = []
                                        cond_cllt[idx] = []
                                    z0_cllt[idx].append(noise[bid].cpu().numpy())
                                    z1_cllt[idx].append(action_pred[bid].cpu().numpy())
                                    cond_cllt[idx].append(cond[bid].cpu().numpy())

            gen_data = {"z0_cllt": z0_cllt, "z1_cllt": z1_cllt, "cond_cllt": cond_cllt}
            with open(f"data/kitchen/{cfg.task.env_name}/{cfg.sefa_name}.pkl", "wb") as f:
                pickle.dump(gen_data, f)
            print(f"Generated RecFlow data saved to {f.name}")
            return

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch and cfg.use_wandb:
                            # log of last step is combined with validation and rollout
                            try:
                                wandb_run.log(step_log, step=self.global_step)
                            except:
                                print(f'wandb error in {self.global_step}, not logged')
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = {'test/mean_score': 0.0, 'test/mean_success_rate': 0.0, 'train/mean_score': 0.0, 'train/mean_success_rate': 0.0}
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        # log
                        step_log['train_action_mse_error'] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if cfg.use_wandb:
                    wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainSeFAUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
