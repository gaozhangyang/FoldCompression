# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, get_args
from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
from megatron.core.distributed import DistributedDataParallelConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.utils.exp_manager import TimingCallback
from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from task.data_interface import ESMDataModule
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.llm.model.biobert.model import BiobertSpecOption
from bionemo.llm.utils.datamodule_utils import float_or_int_or_none, infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig as BionemoWandbConfig, setup_nemo_lightning_logger
from task.model_interface import BionemoLightningModule
from src.utils.utils import process_args
from task.common_args import get_common_parser
from task.custom_args import add_custom_args_to_parser
from task.config import TrainingConfigBundle
import torch
from lightning.pytorch.callbacks import Callback
# from src.utils.callbacks import MyModelCheckpoint
# from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig
import os
os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0"

__all__: Sequence[str] = ("get_parser", "main", "train_esm2_entrypoint")

class ZeroNanGradients(Callback):
    def on_after_backward(self, trainer, pl_module):
        for p in pl_module.parameters():
            if p.grad is not None:
                # 把所有 NaN 和 inf 都设为 0
                p.grad.masked_fill_(torch.isnan(p.grad), 0.0)
                p.grad.masked_fill_(torch.isinf(p.grad), 0.0)

def main(config: TrainingConfigBundle) -> nl.Trainer:
    """Train an ESM2 model on UR data.

    Args:
        config (TrainingConfigBundle): Configuration bundle containing all training parameters
    """
    if config.experiment.infer_feats:
        config.data.data_splits = "0, 0, 1"  # If we are inferring features, we only need the validation split.
        
    # Create the result directory if it does not exist.
    os.makedirs(config.experiment.result_dir, exist_ok=True)

    # Setup the strategy and trainer
    global_batch_size = infer_global_batch_size(
        micro_batch_size=config.training.micro_batch_size,
        num_nodes=config.distributed.num_nodes,
        devices=config.distributed.devices,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        tensor_model_parallel_size=config.distributed.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.distributed.pipeline_model_parallel_size,
    )

    # Initialize the data module with config only.
    data_module = ESMDataModule(config)

    # Set decoder_first_pipeline_num_layers if needed and not provided
    if config.model.num_layers % config.distributed.pipeline_model_parallel_size != 0 and config.distributed.decoder_first_pipeline_num_layers is None:
        config.distributed.decoder_first_pipeline_num_layers = config.model.num_layers - int(config.model.num_layers / config.distributed.pipeline_model_parallel_size + 0.5) * (
            config.distributed.pipeline_model_parallel_size - 1
        )

    

    if config.training.scheduler_num_steps is None:
        config.training.scheduler_num_steps = config.training.num_steps

    model = BionemoLightningModule(config)
    


    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=config.distributed.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.distributed.pipeline_model_parallel_size,
        pipeline_dtype=get_autocast_dtype(config.training.precision),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=config.distributed.overlap_grad_reduce,
            overlap_param_gather=False,
            average_in_collective=config.distributed.average_in_collective,
            grad_reduce_in_fp32=config.distributed.grad_reduce_in_fp32,
            use_distributed_optimizer=True,
        ),
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=False,
        ckpt_parallel_load=True,
        num_layers_in_first_pipeline_stage=config.distributed.decoder_first_pipeline_num_layers,
        # ckpt_load_optimizer=True
    )

    # for wandb integration
    # Please refer to https://pytorch-lightning.readthedocs.io/en/0.7.6/api/lightning.pytorch.loggers.html"
    wandb_config: Optional[BionemoWandbConfig] = (
        None
        if config.wandb.project is None
        else BionemoWandbConfig(
            offline=config.wandb.offline,
            project=config.wandb.project,
            entity=config.wandb.entity,
            tags=config.wandb.tags,
            group=config.wandb.group,
            job_type=config.wandb.job_type,
            id=config.wandb.id,
            anonymous=config.wandb.anonymous,
            log_model=config.wandb.log_model,
        )
    )

    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        nl_callbacks.PreemptionCallback(),
        TimingCallback(),
        # ZeroNanGradients(),
    ]

    if config.profiling.nsys_profiling:
        if config.profiling.nsys_end_step is None:
            config.profiling.nsys_end_step = config.training.num_steps
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=config.profiling.nsys_start_step, 
                end_step=config.profiling.nsys_end_step, 
                ranks=config.profiling.nsys_ranks, 
                gen_shape=True
            )
        )

    # Setup the logger and train the model
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=config.experiment.result_dir,
        name=config.experiment.experiment_name,
        initialize_tensorboard_logger=config.logging.create_tensorboard_logger,
        wandb_config=wandb_config,
    )

    # Configure our custom ModelCheckpointe callback and AutoResume to save at nemo_logger.save_dir/checkpoints
    if config.checkpoint.create_checkpoint_callback:
        checkpoint_path = str(Path(nemo_logger.save_dir) / "checkpoints")
        checkpoint_callback = nl_callbacks.ModelCheckpoint(
            dirpath=checkpoint_path,
            save_last=config.checkpoint.save_last_checkpoint,
            monitor=config.checkpoint.metric_to_monitor_for_checkpoints,  # "val_loss",
            save_top_k=config.checkpoint.save_top_k,
            every_n_train_steps=config.logging.val_check_interval,
            always_save_context=True,
            # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
            filename="{epoch}-{step}-{consumed_samples}",
            # Including step and consumed_samples in the checkpoint filename prevents duplicate filenames and bugs related to this.
            # Save both the weights and the optimizer state.
            save_weights_only=False,
            save_optim_on_train_end=True,
        )
        
        # ckpt_cfg = CheckpointingConfig(
        #     enabled=True,
        #     checkpoint_dir=checkpoint_path,
        #     model_save_format="safetensors",
        #     save_consolidated=False,
        #     model_cache_dir="checkpoints/cache/",
        #     model_repo_id="bionemo/foldcompression",
        #     is_peft=False
        # )
        
        # checkpoint_callback = MyModelCheckpoint(
        #     monitor="val_loss",
        #     save_top_k=5,
        #     save_last=True,
        #     mode="min",
        #     dirpath=checkpoint_path
        # )

        callbacks.append(checkpoint_callback)

        auto_resume = resume.AutoResume(
            resume_from_directory=checkpoint_path,
            resume_if_exists=config.checkpoint.resume_if_exists,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
            resume_past_end=False,
        )
    else:
        auto_resume = None

    trainer = nl.Trainer(
        devices=config.distributed.devices,
        max_steps=config.training.num_steps if config.training.early_stop_on_step is None else config.training.early_stop_on_step,
        accelerator="gpu",
        strategy=strategy,
        limit_val_batches=config.logging.limit_val_batches,  # This controls upsampling and downsampling
        val_check_interval=config.logging.val_check_interval,
        log_every_n_steps=config.logging.log_every_n_steps,
        num_nodes=config.distributed.num_nodes,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(
            precision=config.training.precision,
            params_dtype=get_autocast_dtype(config.training.precision),
            pipeline_dtype=get_autocast_dtype(config.training.precision),
            grad_reduce_in_fp32=config.distributed.grad_reduce_in_fp32,
            autocast_enabled=False,
        ),
        enable_checkpointing=config.checkpoint.create_checkpoint_callback
        # detect_anomaly=True
        # gradient_clip_val=1.0,  # Gradient clipping value
    )
    
    # trainer.custom_ckpt_path = '/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/results/struct_compress/debug/checkpoints/debug--val_loss=1082.3435-epoch=0-consumed_samples=160.0-last.ckpt'
    # trainer.custom_ckpt_path = None

    # trainer.strategy.load_checkpoint('/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/results/struct_compress/baseline_prefix32_len512_dec1/checkpoints/epoch=0-step=99999-consumed_samples=800000.0-last')
    
    if config.experiment.infer_feats:
        llm.validate(
            model=model,
            data=data_module,
            trainer=trainer,
            log=nemo_logger,
            resume=auto_resume,
        )
    else:
        llm.train(
            model=model,
            data=data_module,
            trainer=trainer,
            log=nemo_logger,
            resume=auto_resume,
        )
    return trainer


def train_esm2_entrypoint():
    """Entrypoint for running inference on a geneformer checkpoint and data."""
    # 1. get arguments
    args = get_parsed_args()
    # 2. Create configuration bundle from args
    config = TrainingConfigBundle.from_args(args)
    # 3. Call main with config
    main(config)


def get_parser():
    """Return the cli parser for this tool."""
    # Create parser with common arguments
    parser = get_common_parser(description="Pretrain ESM2 with UR data.")
    
    # Add custom arguments specific to FoldCompression
    parser = add_custom_args_to_parser(parser)
    
    return parser


def get_parsed_args():
    """Return parsed arguments for this tool."""
    parser = get_parser()
    # Process arguments with hydra config
    args = process_args(parser, config_path='../../task/configs')
    print(args)
    return args


if __name__ == "__main__":
    train_esm2_entrypoint()
