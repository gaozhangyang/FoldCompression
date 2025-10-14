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
Custom argument parser for FoldCompression specific parameters.
This module contains project-specific command-line arguments that are unique to the FoldCompression task.
"""

import argparse


def add_custom_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add custom model architecture arguments specific to FoldCompression.
    
    Args:
        parser: The argument parser to add arguments to
        
    Returns:
        The parser with custom model arguments added
    """
    parser.add_argument(
        '--seq_len', 
        default=1024, 
        type=int,
        help="Sequence length for the model. Default is 1024."
    )
    parser.add_argument(
        '--prefix_len', 
        default=6, 
        type=int,
        help="Prefix length for the model. Default is 6."
    )
    parser.add_argument(
        '--enc_layers', 
        default=8, 
        type=int,
        help="Number of encoder layers. Default is 8."
    )
    parser.add_argument(
        '--dec_layers', 
        default=8, 
        type=int,
        help="Number of decoder layers. Default is 8."
    )
    parser.add_argument(
        '--hidden_dim', 
        default=128, 
        type=int,
        help="Hidden dimension size. Default is 128."
    )
    parser.add_argument(
        '--use_dino', 
        default=1, 
        type=int,
        help="Whether to use DINO. Default is False."
    )
    
    return parser


def add_custom_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add custom training arguments specific to FoldCompression.
    
    Args:
        parser: The argument parser to add arguments to
        
    Returns:
        The parser with custom training arguments added
    """
    parser.add_argument(
        '--custom_checkpoint_path', 
        default="", 
        type=str,
        help="Path to custom checkpoint file. Default is empty string."
    )
    parser.add_argument(
        '--infer_feats', 
        default=1, 
        type=int,
        help="Number of inference features. Default is 1."
    )
    parser.add_argument(
        '--nn_neighbors', 
        default=8, 
        type=int,
        help="Number of nearest neighbors. Default is 8."
    )
    parser.add_argument(
        '--input_modality', 
        default=['structure'], 
        type=str,
        nargs='+',
        help="Input modality as a list of strings, e.g. --input_modality structure sequence."
    )
    parser.add_argument(
        '--output_modality', 
        default=['structure', 'sequence'], 
        type=str,
        nargs='+',
        help="Output modality as a list of strings, e.g. --output_modality structure sequence."
    )
    return parser


def get_custom_parser(description: str = "FoldCompression Custom Arguments") -> argparse.ArgumentParser:
    """Create a parser with all custom arguments.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        Argument parser with all custom arguments added
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Add custom argument groups
    parser = add_custom_model_args(parser)
    parser = add_custom_training_args(parser)
    
    return parser


def add_custom_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add custom arguments to an existing parser.
    
    Args:
        parser: The existing argument parser to add custom arguments to
        
    Returns:
        The parser with custom arguments added
    """
    parser = add_custom_model_args(parser)
    parser = add_custom_training_args(parser)
    
    return parser
