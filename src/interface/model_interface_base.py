from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Dict, Union

import lightning.pytorch as pl
import torch
from megatron.core import parallel_state
from nemo.lightning import io as nlio
from nemo.lightning.megatron_parallel import DataT, MegatronLossReduction, ReductionT
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from bionemo.core.model.config import BionemoTrainableModelConfig
from bionemo.llm.api import MegatronLossType, MegatronModelType
# from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

# --- Shared utility functions ---

t = TypeVar('t')

def some_first(seq: Iterable[Optional[t]]) -> t:
    for s in seq:
        if s is not None:
            return s
    raise ValueError("non-None value not found")


def get_dtype_device(torch_object) -> Tuple[torch.dtype, torch.device]:
    # looks up dtype and device recursively
    match torch_object:
        case []:
            raise ValueError("Looking up dtype on an empty list")
        case {**data} if not data:
            raise ValueError("Looking up dtype on an empty dict")
        case torch.Tensor(dtype=dtype, device=device):
            return dtype, device
        case torch.nn.Module() as m:
            try:
                p = next(m.parameters())
            except StopIteration as e:
                raise ValueError("Cannot get dtype on a torch module with no parameters.") from e
            return p.dtype, p.device
        case dict(values=values):
            val = some_first(values())
            return get_dtype_device(val)
        case list() as l:
            val = some_first(l)
            return get_dtype_device(val)
        case _:
            raise TypeError("Got something we didn\'t expect")


def batch_collator(
    batches: Optional[Union[Tuple[ReductionT], List[ReductionT]]],
    batch_dim: int = 0,
    seq_dim: int = 1,
    batch_dim_key_defaults: Dict[str, int] = {"token_logits": 1},
    seq_dim_key_defaults: Dict[str, int] = {"token_logits": 0},
) -> Optional[ReductionT]:
    # ... [same implementation as before] ...
    raise NotImplementedError


class PassthroughLossReduction(MegatronLossReduction, Generic[DataT]):
    """Hijacks Nemo loss reduction to pass through predictions for inference."""

    def forward(self, batch: DataT, forward_out: DataT) -> Tuple[torch.Tensor, DataT]:
        return torch.zeros((1, 1)), forward_out

    def reduce(self, forward_out: List[DataT]) -> DataT:
        return batch_collator(forward_out)


class LightningPassthroughPredictionMixin:
    """Enables inference by passing through forward outputs."""

    def predict_loss_reduction(self) -> PassthroughLossReduction:
        return PassthroughLossReduction()


# --- Abstract base class ---

class ModelInterfaceBase(
    Generic[MegatronModelType, MegatronLossType],
    pl.LightningModule,
    nlio.IOMixin,
    nlio.ConnectorMixin,
    LightningPassthroughPredictionMixin,
    ABC,
):
    """Base LightningModule for BioNemo Megatron models, split into common and user-customizable parts."""

    def __init__(
        self,
        model_transform: Optional[Callable[[MegatronModelType], MegatronModelType]] = None,
        configure_init_model_parallel: bool = False,
        **model_construct_args,
    ) -> None:
        super().__init__()
        self.model_transform = model_transform
        self.configure_init_model_parallel = configure_init_model_parallel
        self.module: Optional[MegatronModelType] = None

    @abstractmethod
    def configure_model(self) -> None:
        """Instantiate `self.module` using `self.config` and stored hparams."""
        ...

    def is_on_logging_device(self) -> bool:
        return (
            parallel_state.is_pipeline_last_stage()
            and parallel_state.get_tensor_model_parallel_rank() == 0
        )

    @abstractmethod
    def data_step(self, dataloader_iter: Iterator[DataT]) -> DataT:
        """Collate a micro-batch from the dataloader."""
        ...

    @abstractmethod
    def forward_step(self, batch: Any) -> Any:
        """Perform the core forward pass, returning model outputs or losses."""
        ...

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: Optional[int] = None) -> Any:
        """Perform the training step, returning  outputs."""
        ... 

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: Optional[int] = None) -> Any:
        """Perform the validation step, returning outputs."""
        ...


    def predict_step(self, batch: Any, batch_idx: Optional[int] = None) -> Any:
        if len(batch) == 0:
            return None
        return self.forward_step(batch)

    def training_loss_reduction(self) -> MegatronLossType:
        return self.loss_reduction_class()

    def validation_loss_reduction(self) -> MegatronLossType:
        return self.loss_reduction_class(validation_step=True)

    def test_loss_reduction(self) -> MegatronLossType:
        return self.loss_reduction_class(validation_step=True)

