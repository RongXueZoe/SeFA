if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
import pickle
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from sefa_policy.workspace.base_workspace import BaseWorkspace
from sefa_policy.dataset.base_dataset import BaseImageDataset
from sefa_policy.common.checkpoint_util import TopKCheckpointManager
from sefa_policy.common.json_logger import JsonLogger
from sefa_policy.common.pytorch_util import dict_apply, optimizer_to
from sefa_policy.model.diffusion.ema_model import EMAModel
from sefa_policy.model.common.lr_scheduler import get_scheduler
from sefa_policy.env_runner.adroit_runner import AdroitRunner
from scripts.find_nearest import find_nearest_action_sequence, load_all_samples

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainSeFAUnetHybridAdroitWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost
        
        # resume training
        if cfg.training.resume and cfg.mode not in ['sefa', 'gen_data']:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        if cfg.mode in ['gen_data', 'sefa']:
            base_ckpt = cfg.base_ckpt
            assert base_ckpt is not None
            sefa_name = cfg.sefa_name
            assert sefa_name is not None
            print(f"Loading base checkpoint from {base_ckpt}")
            self.load_checkpoint(path=base_ckpt)
            if cfg.mode == 'sefa':
                self.epoch = 0
                self.global_step = 0
        else:
            assert cfg.mode in ['train'], f"Invalid mode: {cfg.mode}"

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        # assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        if cfg.mode not in ['sefa', 'gen_data']:
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
        print('lr_scheduler loaded', lr_scheduler)

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)

        if env_runner is not None:
            assert isinstance(env_runner, AdroitRunner)
        
        cfg.logging.name = str(cfg.logging.name)
        if cfg.use_wandb:
            cprint("-----------------------------", "yellow")
            cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
            cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
            cprint("-----------------------------", "yellow")
            # configure logging
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
        print('topk_manager loaded', topk_manager)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        print('optimizer transferred to cuda')

        # save batch for sampling
        train_sampling_batch = None

        if cfg.mode == 'gen_data':
            print(f"Generating SeFA data from {cfg.base_ckpt} with {cfg.sefa_name}")
            self.model.eval()
            z0_cllt = {}
            z1_cllt = {}
            cond_cllt = {}
            with tqdm.tqdm(train_dataloader, desc=f"Generating SeFA data epoch={self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec, dynamic_ncols=True) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True).float())
                    obs_dict = batch["obs"]
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

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
            os.makedirs(f"data/adroit/{cfg.task.name}", exist_ok=True)

            action_sequences, target_poses, img_sequences = load_all_samples(dataset.replay_buffer)

            print(len(z1_cllt), len(z1_cllt[0]))
            for outer_idx in tqdm.tqdm(range(len(z1_cllt)), dynamic_ncols=True, leave=True):
                for inner_idx in tqdm.tqdm(range(len(z1_cllt[outer_idx])), dynamic_ncols=True, leave=False):
                    sefa_action = torch.from_numpy(z1_cllt[outer_idx][inner_idx])    ## [16, 4]
                    sefa_cond = torch.from_numpy(cond_cllt[outer_idx][inner_idx])    ## [16, 4]
                    nearest_sequence = find_nearest_action_sequence(sefa_action, sefa_cond, self.model.obs_encoder, normalizer, action_sequences, target_poses, img_sequences)
                    if nearest_sequence is not None:
                        z1_cllt[outer_idx][inner_idx] = nearest_sequence.cpu().numpy()
            gen_data = {"z0_cllt": z0_cllt, "z1_cllt": z1_cllt, "cond_cllt": cond_cllt}
            with open(f"data/adroit/{cfg.task.name}/{cfg.sefa_name}.pkl", "wb") as f:
                pickle.dump(gen_data, f)
            print(f"Generated refined SeFA data saved to {f.name}")
            return

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                if 'gen_data' in cfg.mode:
                    break
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        t1 = time.time()
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                    
                        # compute loss
                        t1_1 = time.time()
                        raw_loss, loss_dict = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        
                        t1_2 = time.time()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        t1_3 = time.time()
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)
                        t1_4 = time.time()
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
                        t1_5 = time.time()
                        step_log.update(loss_dict)
                        t2 = time.time()
                        
                        if verbose:
                            print(f"total one step time: {t2-t1:.3f}")
                            print(f" compute loss time: {t1_2-t1_1:.3f}")
                            print(f" step optimizer time: {t1_3-t1_2:.3f}")
                            print(f" update ema time: {t1_4-t1_3:.3f}")
                            print(f" logging time: {t1_5-t1_4:.3f}")

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
                if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                    t3 = time.time()
                    # runner_log = env_runner.run(policy, dataset=dataset)
                    if self.epoch > 0:
                        runner_log = env_runner.run(policy)
                    else:
                        runner_log = {'test/mean_score': 0.0, 'test/mean_success_rate': 0.0, 'train/mean_score': 0.0, 'train/mean_success_rate': 0.0}
                    t4 = time.time()
                    # print(f"rollout time: {t4-t3:.3f}")
                    # log all
                    step_log.update(runner_log)

                torch.cuda.empty_cache()
                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss, loss_dict = self.model.compute_loss(batch)
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
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if env_runner is None:
                    step_log['test_mean_score'] = - train_loss
                    
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
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
                del step_log
                # if self.epoch >= cfg.training.num_epochs:  # NOTE: need to set num_epochs in cfg
                #     break

    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        eval_seed = cfg.eval_seed
        torch.manual_seed(eval_seed)
        np.random.seed(eval_seed)
        random.seed(eval_seed)
        method = cfg.method
        
        lastest_ckpt_path = self.get_checkpoint_path(tag=method)
        lastest_ckpt_path = self.get_checkpoint_path(tag='latest')
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        
        # configure env
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, )
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')

    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
