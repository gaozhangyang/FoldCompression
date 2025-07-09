# Copyright Generate Biomedicines, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layers for perturbing protein structure with noise.

This module contains pytorch layers for perturbing protein structure with noise,
which can be useful both for data augmentation, benchmarking, or denoising based
training.
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone, hbonds, rmsd



class ReconstructionLosses(nn.Module):
    """Compute diffusion reconstruction losses for protein backbones.

    Args:
        diffusion (DiffusionChainCov): Diffusion object parameterizing a
            forwards diffusion over protein backbones.
        loss_scale (float): Length scale parameter used for setting loss error
            scaling in units of Angstroms. Default is 10, which corresponds to
            using units of nanometers.
        rmsd_method (str): Method used for computing RMSD superpositions. Can
            be "symeig" (default) or "power" for power iteration.

    Inputs:
        X0_pred (torch.Tensor): Denoised coordinates with shape
            `(num_batch, num_residues, 4, 3)`.
        X (torch.Tensor): Unperturbed coordinates with shape
            `(num_batch, num_residues, 4, 3)`.
        C (torch.LongTensor): Chain map with shape `(num_batch, num_residues)`.
        t (torch.Tensor): Diffusion time with shape `(batch_size,)`.
            Should be on [0,1].

    Outputs:
        losses (dict): Dictionary of reconstructions computed across different
            metrics. Metrics prefixed with `batch_` will be batch-averaged scalars
            while other metrics should be per batch member with shape
            `(num_batch, ...)`.
    """

    def __init__(
        self,
        loss_scale: float = 10.0,
        rmsd_method: str = "symeig",
    ):
        super().__init__()
        self.loss_scale = loss_scale
        self._loss_eps = 1e-5

        # Auxiliary losses
        self.loss_rmsd = rmsd.BackboneRMSD(method=rmsd_method)
        self.loss_fragment = rmsd.LossFragmentRMSD(method=rmsd_method)
        self.loss_fragment_pair = rmsd.LossFragmentPairRMSD(method=rmsd_method)
        self.loss_neighborhood = rmsd.LossNeighborhoodRMSD(method=rmsd_method)
        self.loss_hbond = hbonds.LossBackboneHBonds()
        self.loss_distance = backbone.LossBackboneResidueDistance()

        self.loss_functions = {
            "rmsd": self._loss_rmsd,
            "fragment": self._loss_fragment,
            "pair": self._loss_pair,
            "neighborhood": self._loss_neighborhood,
            "distance": self._loss_distance,
            "hbonds": self._loss_hbonds,
        }

    def _batch_average(self, loss, C):
        weights = (C >= 0).float()
        return (weights * loss).sum() / (weights.sum() + self._loss_eps)


    def _loss_rmsd(self, losses, X0_pred, X, C):
        _, rmsd_denoise = self.loss_rmsd.align(X, X0_pred, C)
        losses["batch_global_mse"] = rmsd_denoise.mean()


    def _loss_fragment(self, losses, X0_pred, X, C):
        # Aligned Fragment MSE loss
        mask = (C >= 0).float()
        rmsd_fragment = self.loss_fragment(X0_pred, X, C)
        losses["batch_fragment_mse"] = self._batch_average(rmsd_fragment, C)

    def _loss_pair(self, losses, X0_pred, X, C):
        # Aligned Pair MSE loss
        rmsd_pair, mask_ij_pair = self.loss_fragment_pair(X0_pred, X, C)
        losses["batch_pair_mse"] = self._batch_average(rmsd_pair.mean(dim=-1), C)

    def _loss_neighborhood(self, losses, X0_pred, X, C):
        # Neighborhood MSE
        rmsd_neighborhood, mask = self.loss_neighborhood(X0_pred, X, C)
        losses["batch_neighborhood_mse"] = self._batch_average(
            rmsd_neighborhood, C
        )

    def _loss_distance(self, losses, X0_pred, X, C):
        # Distance MSE
        mask = (C >= 0).float()
        distance_mse = self.loss_distance(X0_pred, X, C)
        losses["batch_distance_mse"] = self._batch_average(distance_mse, C)

    def _loss_hbonds(self, losses, X0_pred, X, C):
        # HBond recovery
        outs = self.loss_hbond(X0_pred, X, C)
        hb_local, hb_nonlocal, error_co = [o for o in outs]

        losses["batch_hb_local"] = hb_local.mean()
        # losses["hb_local"] = hb_local
        losses["batch_hb_nonlocal"] = hb_nonlocal.mean()
        # losses["hb_nonlocal"] = hb_nonlocal
        losses["batch_hb_contact_order"] = error_co.mean()


    def forward(
        self,
        X0_pred: torch.Tensor,
        X: torch.Tensor,
        C: torch.LongTensor,
    ):
        # Collect all losses and tensors for metric tracking
        losses = {"X": X, "X0_pred": X0_pred}
        for _loss in self.loss_functions.values():
            _loss(losses, X0_pred, X, C)
        return losses
    
    def forward_efficient(
        self,
        X0_pred: torch.Tensor,
        X: torch.Tensor,
        C: torch.LongTensor,
        t: torch.Tensor,
    ):
        # Collect all losses and tensors for metric tracking
        losses = {"t": t, "X": X, "X0_pred": X0_pred}
        for _loss in self.loss_functions.values():
            _loss(losses, X0_pred, X, C, t, w=None, X_t_2=None)
        return losses

