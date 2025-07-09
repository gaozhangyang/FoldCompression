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

"""Layers for building graph representations of protein structure.

This module contains pytorch layers for representing protein structure as a
graph with node and edge features based on geometric information. The graph
features are differentiable with respect to input coordinates and can be used
for building protein scoring functions and optimizing protein geometries
natively in pytorch.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import graph
from . import geometry


class ProteinGraph(nn.Module):
    """Build a graph topology given a protein backbone.

    Args:
        num_neighbors (int): Maximum number of neighbors in the graph.
        distance_atom_type (int): Atom type for computing residue-residue
            distances for graph construction. Negative values will specify
            centroid across atom types. Default is `-1` (centroid).
        cutoff (float): Cutoff distance for graph construction. If not None,
            mask any edges further than this cutoff. Default is `None`.
        mask_interfaces (Boolean): Restrict connections only to within chains,
            excluding-between chain interactions. Default is `False`.
        criterion (string, optional): Method used for building graph from distances.
            Currently supported methods are `{knn, random_log, random_linear}`.
            Default is `knn`.
        random_alpha (float, optional): Length scale parameter for random graph
            generation. Default is 3.
        random_temperature (float, optional): Temperature parameter for
            random graph sampling. Between 0 and 1 this value will interpolate
            between a normal k-NN graph and sampling from the graph generation
            process. Default is 1.0.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, 4, 3)`.
        C (torch.LongTensor): Chain map with shape
            `(num_batch, num_residues)`.
        custom_D (torch.Tensor, optional): Optional external distance map, for example
            based on other distance metrics, with shape
            `(num_batch, num_residues, num_residues)`.
        custom_mask_2D (torch.Tensor, optional): Optional mask to apply to distances
            before computing dissimilarities with shape
            `(num_batch, num_residues, num_residues)`.

    Outputs:
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
                `(num_batch, num_residues, num_neighbors)`.
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.
    """

    def __init__(
        self,
        num_neighbors: int = 30,
        distance_atom_type: int = -1,
        cutoff: Optional[float] = None,
        mask_interfaces: bool = False,
        criterion: str = "knn",
        random_alpha: float = 3.0,
        random_temperature: float = 1.0,
        random_min_local: float = 20,
        deterministic: bool = False,
        deterministic_seed: int = 10,
    ):
        super(ProteinGraph, self).__init__()
        self.num_neighbors = num_neighbors
        self.distance_atom_type = distance_atom_type
        self.cutoff = cutoff
        self.mask_interfaces = mask_interfaces
        self.distances = geometry.Distances()
        self.knn = kNN(k_neighbors=num_neighbors)

        self.criterion = criterion
        self.random_alpha = random_alpha
        self.random_temperature = random_temperature
        self.random_min_local = random_min_local
        self.deterministic = deterministic
        self.deterministic_seed = deterministic_seed

    def _mask_distances(self, X, C, custom_D=None, custom_mask_2D=None):
        mask_1D = chain_map_to_mask(C)
        mask_2D = mask_1D.unsqueeze(2) * mask_1D.unsqueeze(1)
        if self.distance_atom_type > 0:
            X_atom = X[:, :, self.distance_atom_type, :]
        else:
            X_atom = X.mean(dim=2)
        if custom_D is None:
            D = self.distances(X_atom, dim=1)
        else:
            D = custom_D

        if custom_mask_2D is None:
            if self.mask_interfaces:
                mask_2D = torch.eq(C.unsqueeze(1), C.unsqueeze(2))
                mask_2D = mask_2D * mask_2D.type(torch.float32)
            if self.cutoff is not None:
                mask_cutoff = (D <= self.cutoff).type(torch.float32)
                mask_2D = mask_cutoff * mask_2D
        else:
            mask_2D = custom_mask_2D
        return D, mask_1D, mask_2D

    def _perturb_distances(self, D):
        # Replace distance by log-propensity
        if self.criterion == "random_log":
            logp_edge = -3 * torch.log(D)
        elif self.criterion == "random_linear":
            logp_edge = -D / self.random_alpha
        elif self.criterion == "random_uniform":
            logp_edge = D * 0
        else:
            return D

        if not self.deterministic:
            Z = torch.rand_like(D)
        else:
            with torch.random.fork_rng():
                torch.random.manual_seed(self.deterministic_seed)
                Z_shape = [1] + list(D.shape)[1:]
                Z = torch.rand(Z_shape, device=D.device)

        # Sample Gumbel noise
        G = -torch.log(-torch.log(Z))

        # Negate because are doing argmin instead of argmax
        D_key = -(logp_edge / self.random_temperature + G)

        return D_key

    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        custom_D: Optional[torch.Tensor] = None,
        custom_mask_2D: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        D, mask_1D, mask_2D = self._mask_distances(X, C, custom_D, custom_mask_2D)

        if self.criterion != "knn":
            if self.random_min_local > 0:
                # Build first k-NN graph (local)
                self.knn.k_neighbors = self.random_min_local
                edge_idx_local, _, mask_ij_local = self.knn(D, mask_1D, mask_2D)

                # Build mask exluding these first ones
                mask_ij_remaining = 1.0 - mask_ij_local
                mask_2D_remaining = torch.ones_like(mask_2D).scatter(
                    2, edge_idx_local, mask_ij_remaining
                )
                mask_2D = mask_2D * mask_2D_remaining

                # Build second k-NN graph (random)
                self.knn.k_neighbors = self.num_neighbors - self.random_min_local
                D = self._perturb_distances(D)
                edge_idx_random, _, mask_ij_random = self.knn(D, mask_1D, mask_2D)
                edge_idx = torch.cat([edge_idx_local, edge_idx_random], 2)
                mask_ij = torch.cat([mask_ij_local, mask_ij_random], 2)

                # Handle small proteins
                k = min(self.num_neighbors, D.shape[-1])
                edge_idx = edge_idx[:, :, :k]
                mask_ij = mask_ij[:, :, :k]

                self.knn.k_neighbors = self.num_neighbors
                return edge_idx.contiguous(), mask_ij.contiguous()
            else:
                D = self._perturb_distances(D)

        edge_idx, edge_D, mask_ij = self.knn(D, mask_1D, mask_2D)
        return edge_idx, mask_ij


class kNN(nn.Module):
    """Build a k-nearest neighbors graph given a dissimilarity matrix.

    Args:
        k_neighbors (int): Number of nearest neighbors to include as edges of
            each node in the graph.

    Inputs:
        D (torch.Tensor): Dissimilarity matrix with shape
            `(num_batch, num_nodes, num_nodes)`.
        mask (torch.Tensor, optional): Node mask with shape `(num_batch, num_nodes)`.
        mask_2D (torch.Tensor, optional): Edge mask with shape
            `(num_batch, num_nodes, num_nodes)`.

    Outputs:
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, k)`. The slice `edge_idx[b,i,:]` contains
            the indices `{j in N(i)}` of the  k nearest neighbors of node `i`
            in object `b`.
        edge_D (torch.Tensor): Distances to each neighbor with shape
            `(num_batch, num_nodes, k)`.
        mask_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
    """

    def __init__(self, k_neighbors: int):
        super(kNN, self).__init__()
        self.k_neighbors = k_neighbors

    def forward(
        self,
        D: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_2D: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        mask_full = None
        if mask is not None:
            mask_full = mask.unsqueeze(2) * mask.unsqueeze(1)
        if mask_2D is not None:
            mask_full = mask_2D if mask_full is None else mask_full * mask_2D
        if mask_full is not None:
            max_float = np.finfo(np.float32).max
            D = mask_full * D + (1.0 - mask_full) * max_float

        k = min(self.k_neighbors, D.shape[-1])
        edge_D, edge_idx = torch.topk(D, int(k), dim=-1, largest=False)

        mask_ij = None
        if mask_full is not None:
            mask_ij = graph.collect_edges(mask_full.unsqueeze(-1), edge_idx)
            mask_ij = mask_ij.squeeze(-1)
        return edge_idx, edge_D, mask_ij

def chain_map_to_mask(C: torch.LongTensor) -> torch.Tensor:
    """Convert chain map into a mask.

    Args:
        C (torch.LongTensor): Chain map with shape
            `(num_batch, num_residues)`.

    Returns:
        mask (Tensor, optional): Mask tensor with shape
            `(num_batch, num_residues)`.
    """
    return (C > 0).type(C.dtype)

