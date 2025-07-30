from nemo.lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint as BaseCKPT
# from nemo_automodel.components.checkpoint.checkpointing import save_model, load_model, save_optimizer, load_optimizer, CheckpointingConfig
from nemo.utils.get_rank import is_global_rank_zero
import torch
import os
from lightning.pytorch.trainer import call
from nemo.utils import logging
import shutil

def unwrap_model(model):
    """
    Recursively unwrap DDP, Float16Module, or any wrapper with `.module`
    """
    while hasattr(model, "module"):
        model = model.module
    return model

class MyModelCheckpoint(BaseCKPT):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        # self.ckpt_cfg = ckpt_cfg

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        optim = trainer.optimizers[0].mcore_optimizer.chained_optimizers[0]
        optim_param_state = optim.get_parameter_state_dp_zero()
        torch.distributed.barrier()
        if is_global_rank_zero():

            model = trainer.lightning_module

            # 保存模型权重
            # save_model(, filepath, self.ckpt_cfg)

            # 保存 optimizer/scheduler 状态
            
            
            # optim.mcore_optimizer.get_parameter_state_dp_zero()
            # optim.mcore_optimizer.chained_optimizers[0].get_parameter_state_dp_zero()
            

            optim_state = {'optim_param_state':optim_param_state, 
                        'optim_state': optim.state_dict(),
                        'global_step': trainer.global_step,
                        'epoch': trainer.current_epoch,
                        'state_dict': unwrap_model(model).state_dict(),
                        'loops': {'fit_loop': trainer.fit_loop.state_dict(),
                                    'validate_loop': trainer.validate_loop.state_dict(),
                                    'test_loop': trainer.test_loop.state_dict(),
                                    'predict_loop': trainer.predict_loop.state_dict()},
                        'lr_schedulers': [trainer.lightning_module.lr_schedulers().state_dict()],
                        'callback_states': {
                                            f"{type(cb).__name__}": cb.state_dict()
                                            for i, cb in enumerate(trainer.callbacks)}
                        }
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(optim_state, filepath)
            
            self.deferred_ckpts_to_remove.append([])
            
            # ===== Top-K Checkpoint 维护逻辑 =====
            monitor_candidates = trainer.callback_metrics
            current_score = float(monitor_candidates[self.monitor].detach().cpu().item())
            
            if current_score is not None:
                self.best_k_models[filepath] = current_score

                # 如果超出 top_k，就移除最差的 checkpoint
                if len(self.best_k_models) > self.save_top_k:
                    reverse = self.mode != "min"
                    worst_ckpt = min(self.best_k_models, key=self.best_k_models.get) if not reverse else max(self.best_k_models, key=self.best_k_models.get)
                    self._remove_checkpoint(trainer, worst_ckpt)

                # 更新 best_model_path 等信息
                sorted_ckpts = sorted(self.best_k_models, key=self.best_k_models.get, reverse=(self.mode != "min"))
                self.best_model_path = sorted_ckpts[0]
                self.best_model_score = self.best_k_models[self.best_model_path]
                self.kth_best_model_path = sorted_ckpts[-1] if sorted_ckpts else ""

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            # Lightning 的默认 checkpoint 行为（保存 epoch, global_step, callbacks, loops, etc.）
            
            # optim.state[optim.param_groups[0]['params'][2]]['exp_avg'].max()

    def on_fit_start(self, trainer, pl_module):
        ckpt_path = trainer.custom_ckpt_path
        if (ckpt_path is not None) and os.path.exists(ckpt_path):
            data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            optim = trainer.optimizers[0].mcore_optimizer.chained_optimizers[0]
            optim.load_parameter_state_from_dp_zero(data['optim_param_state'])
            torch.distributed.barrier()
            
        if trainer.custom_ckpt_path:
            
            if is_global_rank_zero() and os.path.exists(ckpt_path):
                # restore model state
                
                unwrap_model(pl_module).load_state_dict(data['state_dict'])
                

                # restore optimizer state
                # optim = trainer.optimizers[0]
                
                
                optim.load_state_dict(data['optim_state'])
                
                # restore callbacks state
                prec_plugin = trainer.precision_plugin
                prec_plugin.on_load_checkpoint(data)
                
                call._call_callbacks_load_state_dict(trainer, data)
                
                # restore training state 
                # if prec_plugin.__class__.__qualname__ in data:
                #     prec_plugin.load_state_dict(data[prec_plugin.__class__.__qualname__])
            
            
                trainer.fit_loop.load_state_dict(data["loops"]["fit_loop"])
                for idx, callback in enumerate(trainer.callbacks):
                    trainer.callbacks[idx].load_state_dict(data["callback_states"][f"{type(callback).__name__}"])
                
                lr_schedulers = data["lr_schedulers"]
                for config, lrs_state in zip(trainer.lr_scheduler_configs, lr_schedulers):
                    config.scheduler.load_state_dict(lrs_state)
                    
                # restore data module state
                trainer.datamodule.update_init_global_step()
                # trainer.datamodule.load_state_dict(consumed_samples)
                
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
    
    def _remove_checkpoint(self, trainer, filepath: str, override_async=False) -> None:
        from nemo.utils.get_rank import is_global_rank_zero

        if not is_global_rank_zero():
            return

        # 如果存在于 best_k_models 中，先移除
        if filepath in self.best_k_models:
            self.best_k_models.pop(filepath)

        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logging.info(f"Removed checkpoint: {filepath}")
            ckpt_dir = filepath.replace('.ckpt', '')
            if os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir, ignore_errors=True)
        except Exception as e:
            logging.warning(f"Failed to remove checkpoint {filepath}: {e}")

