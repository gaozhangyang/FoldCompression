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

"""
Configuration classes for organizing training parameters.
This module contains dataclasses to organize the many parameters used in training.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.llm.model.biobert.model import BiobertSpecOption


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    cluster_path: Path
    database_path: Path
    min_seq_length: Optional[int]
    max_seq_length: int
    num_dataset_workers: int
    data_splits: str = '95, 4, 1'
    random_mask_strategy: RandomMaskStrategy = RandomMaskStrategy.ALL_TOKENS


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    num_steps: int
    scheduler_num_steps: Optional[int]
    warmup_steps: int
    lr: float
    micro_batch_size: int
    accumulate_grad_batches: int
    precision: PrecisionTypes
    early_stop_on_step: Optional[int] = None


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    biobert_spec_option: BiobertSpecOption
    num_layers: int = 33
    hidden_size: int = 1280
    num_attention_heads: int = 20
    ffn_hidden_size: int = 1280 * 4
    # Custom model parameters
    prefix_len: int = 6
    enc_layers: int = 8
    dec_layers: int = 8
    hidden_dim: int = 128
    nn_neighbors: int = 10
    input_modality: List[str] = None
    output_modality: List[str] = None
    use_dino: bool = False
    
    def __post_init__(self):
        if self.input_modality is None:
            self.input_modality = ['structure', 'sequence']
        if self.output_modality is None:
            self.output_modality = ['structure', 'sequence']


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    num_nodes: int
    devices: int
    pipeline_model_parallel_size: int = 1
    tensor_model_parallel_size: int = 1
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True
    average_in_collective: bool = True
    grad_reduce_in_fp32: bool = False
    decoder_first_pipeline_num_layers: Optional[int] = None


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    limit_val_batches: int
    val_check_interval: int
    log_every_n_steps: Optional[int]
    create_tensorboard_logger: bool = False
    create_tflops_callback: bool = True


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    entity: Optional[str] = None
    project: Optional[str] = None
    offline: bool = False
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    id: Optional[str] = None
    anonymous: Optional[bool] = False
    log_model: bool = False


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    create_checkpoint_callback: bool = True
    save_best_checkpoint: bool = True
    save_last_checkpoint: bool = True
    metric_to_monitor_for_checkpoints: str = "val_loss"
    save_top_k: int = 2
    resume_if_exists: bool = False
    nemo1_init_path: Optional[Path] = None
    restore_from_checkpoint_path: Optional[str] = None


@dataclass
class ProfilingConfig:
    """Configuration for profiling and debugging."""
    nsys_profiling: bool = False
    nsys_start_step: int = 0
    nsys_end_step: Optional[int] = None
    nsys_ranks: List[int] = None
    
    def __post_init__(self):
        if self.nsys_ranks is None:
            self.nsys_ranks = [0]


@dataclass
class ExperimentConfig:
    """Configuration for experiment management."""
    result_dir: str
    experiment_name: str
    infer_feats: int = 0
    custom_checkpoint_path: str = ""


@dataclass
class TrainingConfigBundle:
    """Bundle of all configuration classes."""
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    distributed: DistributedConfig
    logging: LoggingConfig
    wandb: WandbConfig
    checkpoint: CheckpointConfig
    profiling: ProfilingConfig
    experiment: ExperimentConfig

    @classmethod
    def from_args(cls, args):
        """Create configuration bundle from parsed arguments."""
        return cls(
            data=DataConfig(
                cluster_path=args.cluster_path,
                database_path=args.database_path,
                min_seq_length=args.min_seq_length,
                max_seq_length=args.max_seq_length,
                num_dataset_workers=args.num_dataset_workers,
                data_splits=args.data_splits,
                random_mask_strategy=args.random_mask_strategy,
            ),
            training=TrainingConfig(
                num_steps=args.num_steps,
                scheduler_num_steps=args.scheduler_num_steps,
                warmup_steps=args.warmup_steps,
                lr=args.lr,
                micro_batch_size=args.micro_batch_size,
                accumulate_grad_batches=args.accumulate_grad_batches,
                precision=args.precision,
                early_stop_on_step=args.early_stop_on_step,
            ),
            model=ModelConfig(
                biobert_spec_option=args.biobert_spec_option,
                num_layers=args.num_layers,
                hidden_size=args.hidden_size,
                num_attention_heads=args.num_attention_heads,
                ffn_hidden_size=args.ffn_hidden_size,
                prefix_len=args.prefix_len,
                enc_layers=args.enc_layers,
                dec_layers=args.dec_layers,
                hidden_dim=args.hidden_dim,
                nn_neighbors=args.nn_neighbors,
                input_modality=args.input_modality,
                output_modality=args.output_modality,
                use_dino=args.use_dino,
            ),
            distributed=DistributedConfig(
                num_nodes=args.num_nodes,
                devices=args.num_gpus,
                pipeline_model_parallel_size=args.pipeline_model_parallel_size,
                tensor_model_parallel_size=args.tensor_model_parallel_size,
                overlap_grad_reduce=not args.no_overlap_grad_reduce,
                overlap_param_gather=not args.no_overlap_param_gather,
                average_in_collective=not args.no_average_in_collective,
                grad_reduce_in_fp32=args.grad_reduce_in_fp32,
                decoder_first_pipeline_num_layers=args.decoder_first_pipeline_num_layers,
            ),
            logging=LoggingConfig(
                limit_val_batches=args.limit_val_batches,
                val_check_interval=args.val_check_interval,
                log_every_n_steps=args.log_every_n_steps,
                create_tensorboard_logger=args.create_tensorboard_logger,
                create_tflops_callback=args.create_tflops_callback,
            ),
            wandb=WandbConfig(
                entity=args.wandb_entity,
                project=args.wandb_project,
                offline=args.wandb_offline,
                tags=args.wandb_tags,
                group=args.wandb_group,
                job_type=args.wandb_job_type,
                id=args.wandb_id,
                anonymous=args.wandb_anonymous,
                log_model=args.wandb_log_model,
            ),
            checkpoint=CheckpointConfig(
                create_checkpoint_callback=args.create_checkpoint_callback,
                save_best_checkpoint=args.save_best_checkpoint,
                save_last_checkpoint=args.save_last_checkpoint,
                metric_to_monitor_for_checkpoints=args.metric_to_monitor_for_checkpoints,
                save_top_k=args.save_top_k,
                resume_if_exists=args.resume_if_exists,
                nemo1_init_path=args.nemo1_init_path,
                restore_from_checkpoint_path=args.restore_from_checkpoint_path,
            ),
            profiling=ProfilingConfig(
                nsys_profiling=args.nsys_profiling,
                nsys_start_step=args.nsys_start_step,
                nsys_end_step=args.nsys_end_step,
                nsys_ranks=args.nsys_ranks,
            ),
            experiment=ExperimentConfig(
                result_dir=args.result_dir,
                experiment_name=args.experiment_name,
                infer_feats=args.infer_feats,
                custom_checkpoint_path=args.custom_checkpoint_path,
            ),
        )
