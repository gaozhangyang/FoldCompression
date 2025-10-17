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
Common argument parser for training tasks.
This module contains shared command-line arguments that are commonly used across different training scripts.
"""

import argparse
from pathlib import Path
from typing import get_args
from bionemo.core.utils.dtypes import PrecisionTypes
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.llm.model.biobert.model import BiobertSpecOption
from bionemo.llm.utils.datamodule_utils import float_or_int_or_none


def add_common_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common training arguments to the parser.
    
    Args:
        parser: The argument parser to add arguments to
        
    Returns:
        The parser with common training arguments added
    """
    # Data paths
    parser.add_argument(
        "--cluster-path",
        type=str,
        required=False,
        help="Path to the train cluster data parquet file",
    )
    parser.add_argument(
        "--database-path",
        type=str,
        required=False,
        help="Path to the train sequence database file",
    )
    
    # Training configuration
    parser.add_argument(
        "--precision",
        type=str,
        choices=get_args(PrecisionTypes),
        required=False,
        default="bf16-mixed",
        help="Precision type to use for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=4e-4,
        help="Learning rate for training. Default is 4e-4",
    )
    parser.add_argument(
        "--scheduler-num-steps",
        type=int,
        required=False,
        help="Number of steps for learning rate scheduler. Will use --num-steps if not given. Default is None.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=False,
        default=500000,
        help="Number of steps to use for training. Default is 500000.",
    )
    parser.add_argument(
        "--early-stop-on-step",
        type=int,
        default=None,
        help="Stop training on this step, if set. This may be useful for testing or debugging purposes.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        required=False,
        default=2000,
        help="Number of warmup steps for WarmupAnnealDecayHold Scheduler. Default is 2000.",
    )
    
    # Batch and data configuration
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        required=False,
        default=64,
        help="Micro-batch size. Global batch size is inferred from this.",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        required=False,
        default=1,
        help="Gradient accumulation steps. Global batch size is inferred from this.",
    )
    parser.add_argument(
        "--min-seq-length",
        type=float_or_int_or_none,
        required=False,
        default=1024,
        help="Minimum sequence length. Sampled will be padded if less than this value. Set 'None' to unset minimum.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        required=False,
        default=1024,
        help="Maximum sequence length. Samples will be truncated if exceeds this value.",
    )
    parser.add_argument(
        "--num-dataset-workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers to use for training. Default is 1.",
    )
    parser.add_argument(
        "--data-splits",
        type=str,
        required=False,
        default='95, 4, 1',
        help="Data splits for train, validation and test sets. Default is '95, 4, 1'"
    )
    
    return parser


def add_distributed_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add distributed training arguments to the parser.
    
    Args:
        parser: The argument parser to add arguments to
        
    Returns:
        The parser with distributed training arguments added
    """
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=False,
        default=1,
        help="Number of GPUs to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=False,
        default=1,
        help="Number of nodes to use for training. Default is 1.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="Pipeline model parallel size. Default is 1.",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="Tensor model parallel size. Default is 1.",
    )
    
    # DDP configuration
    parser.add_argument(
        "--no-overlap-grad-reduce",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-overlap-param-gather",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-average-in-collective",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--grad-reduce-in-fp32",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--decoder-first-pipeline-num-layers",
        type=int,
        required=False,
        default=None,
        help="The number of transformer layers on the first pipeline stage of the decoder. Default None is even split of transformer layers across all pipeline stages",
    )
    
    return parser


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add model architecture arguments to the parser.
    
    Args:
        parser: The argument parser to add arguments to
        
    Returns:
        The parser with model arguments added
    """
    parser.add_argument(
        "--biobert-spec-option",
        type=BiobertSpecOption,
        choices=[e.value for e in BiobertSpecOption],
        required=False,
        default=BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec.value,
        help="Biobert spec option to use for the model. Default is 'esm2_bert_layer_with_transformer_engine_spec'.",
    )
    parser.add_argument(
        "--random-mask-strategy",
        type=RandomMaskStrategy,
        choices=[e.value for e in RandomMaskStrategy],
        default=RandomMaskStrategy.ALL_TOKENS.value,
        help=f"""In ESM2 pretraining, 15%% of all tokens are masked and among which 10%% are replaced with a random token. This class controls the set of random tokens to choose from. Options are: '{"', '".join([e.value for e in RandomMaskStrategy])}'. Note that 'all_token' will introduce non-canonical amino acid tokens as effective mask tokens, and the resultant loss will appear lower than that from 'amino_acids_only'. Note that 'all_token' is the method used in hugging face as well as portions of fairseq.""",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        required=False,
        default=33,
        help="Number of layers in the model. Default is 33.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        required=False,
        default=1280,
        help="Hidden size of the model. Default is 1280.",
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        required=False,
        default=20,
        help="Number of attention heads in the model. Default is 20.",
    )
    parser.add_argument(
        "--ffn-hidden-size",
        type=int,
        required=False,
        default=4 * 1280,
        help="FFN hidden size of the model. Default is 4 * 1280.",
    )
    
    return parser


def add_logging_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add logging and monitoring arguments to the parser.
    
    Args:
        parser: The argument parser to add arguments to
        
    Returns:
        The parser with logging arguments added
    """
    parser.add_argument(
        "--create-tflops-callback",
        action="store_true",
        default=False,
        help="Enable tflops calculation callback for Hyena / Evo2. Defaults to False.",
    )
    parser.add_argument(
        "--create-tensorboard-logger", 
        action="store_true", 
        default=False, 
        help="Create a tensorboard logger."
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        required=False,
        default=10000,
        help="Number of steps between validation. Default is 10000.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        required=False,
        help="Number of steps between logging. Default is 50.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float_or_int_or_none,
        required=False,
        default=2,
        help="Number of global batches used for validation if int. Fraction of validation dataset if float. Default is 2.",
    )
    
    # Wandb configuration
    parser.add_argument("--wandb-entity", type=str, default=None, help="The team posting this run")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name ")
    parser.add_argument("--wandb-tags", nargs="+", type=str, default=None, help="Tags associated with this run")
    parser.add_argument(
        "--wandb-group", type=str, default=None, help="A unique string shared by all runs in a given group"
    )
    parser.add_argument(
        "--wandb-job-type",
        type=str,
        default=None,
        help="A unique string representing a type of run, which is useful when you're grouping runs together into larger experiments using group.",
    )
    parser.add_argument(
        "--wandb-id", type=str, default=None, help="Sets the version, mainly used to resume a previous run"
    )
    parser.add_argument(
        "--wandb-anonymous", action="store_true", help="Enable or explicitly disable anonymous logging"
    )
    parser.add_argument(
        "--wandb-log-model", action="store_true", help="Save checkpoints in wandb dir to upload on W&B servers"
    )
    parser.add_argument("--wandb-offline", help="Use wandb in offline mode")
    
    return parser


def add_checkpoint_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add checkpoint and resumption arguments to the parser.
    
    Args:
        parser: The argument parser to add arguments to
        
    Returns:
        The parser with checkpoint arguments added
    """
    parser.add_argument(
        "--resume-if-exists", 
        action="store_true", 
        default=False, 
        help="Resume training if a checkpoint exists."
    )
    parser.add_argument(
        "--nemo1-init-path",
        type=Path,
        required=False,
        help="Path to nemo1 file, if desired to load at init time.",
    )
    parser.add_argument(
        "--disable-checkpointing",
        action="store_false",
        default=True,
        dest="create_checkpoint_callback",
        help="Disable creating a ModelCheckpoint callback.",
    )
    parser.add_argument(
        "--save-best-checkpoint",
        action="store_true",
        default=True,
        help="Save the best checkpoint based on the metric to monitor.",
    )
    parser.add_argument(
        "--no-save-best-checkpoint",
        action="store_false",
        default=True,
        dest="save_best_checkpoint",
        help="Disable saving the best checkpoint based on the metric to monitor.",
    )
    parser.add_argument(
        "--save-last-checkpoint",
        action="store_true",
        default=True,
        help="Save the last checkpoint.",
    )
    parser.add_argument(
        "--no-save-last-checkpoint",
        action="store_false",
        dest="save_last_checkpoint",
        default=True,
        help="Disable saving the last checkpoint.",
    )
    parser.add_argument(
        "--metric-to-monitor-for-checkpoints",
        type=str,
        required=False,
        default="val_loss",
        help="The metric to monitor for checkpointing.",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        required=False,
        default=2,
        help="Save the top k checkpoints.",
    )
    parser.add_argument(
        "--restore-from-checkpoint-path",
        type=Path,
        required=False,
        default=None,
        help="Path to the checkpoint directory to restore from. Will override `--resume-if-exists` when set.",
    )
    
    return parser


def add_profiling_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add profiling and debugging arguments to the parser.
    
    Args:
        parser: The argument parser to add arguments to
        
    Returns:
        The parser with profiling arguments added
    """
    parser.add_argument(
        "--nsys-profiling",
        action="store_true",
        default=False,
        help="Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling output you must run the whole program with `nsys`. For example: "
        " `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop  [regular python command here]`",
    )
    parser.add_argument(
        "--nsys-start-step",
        type=int,
        required=False,
        default=0,
        help="Start nsys profiling after this step.",
    )
    parser.add_argument(
        "--nsys-end-step",
        type=int,
        required=False,
        help="End nsys profiling after this step.",
    )
    parser.add_argument(
        "--nsys-ranks",
        type=int,
        nargs="+",
        required=False,
        default=[0],
        help="Enable nsys profiling for these ranks.",
    )
    
    return parser


def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add experiment and output arguments to the parser.
    
    Args:
        parser: The argument parser to add arguments to
        
    Returns:
        The parser with experiment arguments added
    """
    parser.add_argument(
        "--result-dir", 
        type=str, 
        required=False, 
        default="./results", 
        help="Path to the result directory."
    )
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        required=False, 
        default="esm2", 
        help="Name of the experiment."
    )
    parser.add_argument(
        "--config_name", 
        type=str, 
        default='baseline', 
        help="Name of the Hydra config to use"
    )
    
    parser.add_argument(
        "--eval-every-n-steps",
        type=int,
        required=False,
        default=5000,
        help="Evaluate every n steps. Default is 0.",
    )
    parser.add_argument(
        "--eval-data-root",
        type=str,
        required=False,
        default="/mnt/shared-storage-user/gaozhangyang/workspace/FoldCompression/homology_analysis/data/ASTRAL40_pdbstyle/pdbstyle-2.08/",
        help="Path to the evaluation data root directory.",
    )
    parser.add_argument(
        "--eval-labels-tsv",
        type=str,
        required=False,
        default="/mnt/shared-storage-user/gaozhangyang/workspace/FoldCompression/homology_analysis/data/ASTRAL40_pdbstyle/labels.tsv",
        help="Path to the evaluation labels.tsv file.",
    )
    
    
    return parser


def get_common_parser(description: str = "Training script") -> argparse.ArgumentParser:
    """Create a parser with all common arguments.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        Argument parser with all common arguments added
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Add all common argument groups
    parser = add_common_training_args(parser)
    parser = add_distributed_training_args(parser)
    parser = add_model_args(parser)
    parser = add_logging_args(parser)
    parser = add_checkpoint_args(parser)
    parser = add_profiling_args(parser)
    parser = add_experiment_args(parser)
    
    return parser
