import torch
import torch.nn as nn
from foldtoken.model.chroma import rmsd
from . import hbonds
from . import backbone
from .transforms import transform_flatten, transform_cbach_to_sbatch

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

        # self.loss_functions = {
        #     "rmsd": self._loss_rmsd,
        #     "fragment": self._loss_fragment,
        #     "pair": self._loss_pair,
        #     "neighborhood": self._loss_neighborhood,
        #     "distance": self._loss_distance,
        #     "hbonds": self._loss_hbonds,
        # }

    def _batch_average(self, loss, C):
        weights = (C > 0).float()
        return (weights * loss).sum() / (weights.sum() + self._loss_eps)


    def _loss_rmsd(self, losses, X0_pred, X, C):
        rmsd_denoise = self.loss_rmsd.align(X0_pred,X, C)
        losses["batch_global_mse"] = rmsd_denoise[~torch.isnan(rmsd_denoise)].mean()


    def _loss_fragment(self, losses, X0_pred, X, C):
        # Aligned Fragment MSE loss
        rmsd_fragment = self.loss_fragment(X0_pred, X, C)
        losses["batch_fragment_mse"] = rmsd_fragment[~torch.isnan(rmsd_fragment)].mean()

    def _loss_pair(self, losses, X0_pred, X, C):
        # Aligned Pair MSE loss
        rmsd_pair = self.loss_fragment_pair(X0_pred, X, C)
        losses["batch_pair_mse"] = rmsd_pair[~torch.isnan(rmsd_pair)].mean()

    def _loss_neighborhood(self, losses, X0_pred, X, C):
        # Neighborhood MSE
        rmsd_neighborhood = self.loss_neighborhood(X0_pred, X, C)
        losses["batch_neighborhood_mse"] = rmsd_neighborhood[~torch.isnan(rmsd_neighborhood)].mean()

    def _loss_distance(self, losses, X0_pred, X, C):
        # Distance MSE
        _, X0_pred, _ = transform_cbach_to_sbatch(C, X0_pred)
        C_batch, X, mask_batch = transform_cbach_to_sbatch(C, X)
        X0_pred = X0_pred.reshape(*C_batch.shape,4,3)
        X = X.reshape(*C_batch.shape,4,3)
        distance_mse = self.loss_distance(X0_pred, X, C_batch)
        losses["batch_distance_mse"] = distance_mse.mean()

    def _loss_hbonds(self, losses, X0_pred, X, C):
        C_batch, X, mask_batch = transform_cbach_to_sbatch(C, X)
        _, X0_pred, _ = transform_cbach_to_sbatch(C, X0_pred)
        B = X0_pred.shape[0]
        X0_pred = X0_pred.reshape(B,-1,4,3)
        X = X.reshape(B,-1,4,3)
        # HBond recovery
        outs = self.loss_hbond(X0_pred, X, C_batch)
        hb_local, hb_nonlocal, error_co = [o for o in outs]

        losses["batch_hb_local"] = hb_local.mean()
        # losses["hb_local"] = hb_local
        losses["batch_hb_nonlocal"] = hb_nonlocal.mean()
        # losses["hb_nonlocal"] = hb_nonlocal
        losses["batch_hb_contact_order"] = error_co.mean()

        self.loss_functions = {
            "rmsd": self._loss_rmsd,
            "fragment": self._loss_fragment,
            "pair": self._loss_pair,
            "neighborhood": self._loss_neighborhood,
            "distance": self._loss_distance,
            "hbonds": self._loss_hbonds,
        }
        
    def forward(
        self,
        X0_pred: torch.Tensor,
        X: torch.Tensor,
        C: torch.LongTensor,
    ):
        # Collect all losses and tensors for metric tracking
        losses = {"X": X, "X0_pred": X0_pred}
        self._loss_rmsd(losses, X0_pred, X, C)
        # self._loss_fragment(losses, X0_pred, X, C)
        # self._loss_pair(losses, X0_pred, X, C)
        # self._loss_neighborhood(losses, X0_pred, X, C)
        # self._loss_distance(losses, X0_pred, X, C)
        # self._loss_hbonds(losses, X0_pred, X, C)
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

