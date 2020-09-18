import torch
from typing import NamedTuple


class DIST(NamedTuple):
    sample: torch.Tensor
    kl: torch.Tensor


class ZZ(NamedTuple):
    mu: torch.Tensor
    std: torch.Tensor


class MetricMiniBatchSimple(NamedTuple):
    loss: torch.Tensor
    mse: torch.Tensor
    kl_tot: torch.Tensor
    geco_balance: torch.Tensor
    delta_2: torch.Tensor

    def pretty_print(self, epoch: int=0) -> str:
        s = '[epoch {0:4d}] loss={1:.3f}, mse={2:.3f}, kl_tot={3:.3f}, geco_bal={4:.3f}'.format(epoch,
                                                                                                self.loss.item(),
                                                                                                self.mse.item(),
                                                                                                self.kl_tot.item(),
                                                                                                self.geco_balance.item(),
                                                                                                self.delta_2.item())
        return s


class OutputSimple(NamedTuple):
    metrics: MetricMiniBatchSimple
    z: torch.Tensor
    imgs: torch.Tensor
