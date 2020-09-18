import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .namedtuple import *
from .utilities import *


def create_ckpt(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                hyperparams_dict: dict,
                epoch: int) -> dict:

    all_member_var = model.__dict__
    member_var_to_save = {}
    for k, v in all_member_var.items():
        if not k.startswith("_") and k != 'training':
            member_var_to_save[k] = v

    ckpt = {'epoch': epoch,
            'model_member_var': member_var_to_save,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyperparam_dict': hyperparams_dict}

    return ckpt


def ckpt2file(ckpt: dict, path: str):
    torch.save(ckpt, path)


def file2ckpt(path: str, device: Optional[str]=None):
    """ wrapper around torch.load """
    if device is None:
        ckpt = torch.load(path)
    elif device == 'cuda':
        ckpt = torch.load(path, map_location="cuda:0")
    elif device == 'cpu':
        ckpt = torch.load(path, map_location=torch.device('cpu'))
    else:
        raise Exception("device is not recognized")
    return ckpt


def load_from_ckpt(ckpt,
                   model: Optional[torch.nn.Module] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   overwrite_member_var: bool = False):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if model is not None:

        # load member variables
        if overwrite_member_var:
            for key, value in ckpt['model_member_var'].items():
                setattr(model, key, value)

        # load the modules
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])


def instantiate_optimizer(model: torch.nn.Module,
                          dict_params_optimizer: dict) -> torch.optim.Optimizer:
    
    # split the parameters between GECO and NOT_GECO
    geco_params, other_params = [], []
    for name, param in model.named_parameters():
        if name.startswith("geco"):
            geco_params.append(param)
        else:
            other_params.append(param)

    if dict_params_optimizer["type"] == "adam":
        optimizer = torch.optim.Adam([{'params': geco_params, 'lr': dict_params_optimizer["base_lr_geco"],
                                       'betas': dict_params_optimizer["betas_geco"]},
                                      {'params': other_params, 'lr': dict_params_optimizer["base_lr"],
                                       'betas': dict_params_optimizer["betas"]}],
                                     eps=dict_params_optimizer["eps"],
                                     weight_decay=dict_params_optimizer["weight_decay"])
        
    elif dict_params_optimizer["type"] == "SGD":
        optimizer = torch.optim.SGD([{'params': geco_params, 'lr': dict_params_optimizer["base_lr_geco"]},
                                     {'params': other_params, 'lr': dict_params_optimizer["base_lr"]}],
                                    weight_decay=dict_params_optimizer["weight_decay"])
    else:
        raise Exception
    return optimizer


def instantiate_scheduler(optimizer: torch.optim.Optimizer,
                          dict_params_scheduler: dict) -> torch.optim.lr_scheduler:
    if dict_params_scheduler["scheduler_type"] == "step_LR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=dict_params_scheduler["scheduler_step_size"],
                                                    gamma=dict_params_scheduler["scheduler_gamma"],
                                                    last_epoch=-1)
    else:
        raise Exception
    return scheduler





EPS_STD = 1E-3  # standard_deviation = F.softplus(x) + EPS_STD >= EPS_STD

class DecoderConv(nn.Module):
    """ Decode z -> x
        INPUT:  z of shape: ..., dim_z
        OUTPUT: image of shape: ..., ch_out, width, height
        where ... are all the independent dimensions, i.e. box, batch_size, enumeration_dim etc.
    """

    def __init__(self, size: int, dim_z: int, ch_out: int):
        super().__init__()
        self.width = size
        assert self.width == 28
        self.dim_z: int = dim_z
        self.ch_out: int = ch_out
        self.upsample = nn.Linear(self.dim_z, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, self.ch_out, 4, 1, 2)  # B, ch, 28, 28
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        independent_dim = list(z.shape[:-1])
        x1 = self.upsample(z.view(-1, self.dim_z)).view(-1, 64, 7, 7)
        return self.decoder(x1).view(independent_dim + [self.ch_out, self.width, self.width])


class EncoderConv(nn.Module):
    """ Encode x -> z_mu, z_std
        INPUT  x of shape: ..., ch_raw_image, width, height
        OUTPUT z_mu, z_std of shape: ..., latent_dim
        where ... are all the independent dimensions, i.e. box, batch_size, enumeration_dim etc.
    """

    def __init__(self, size: int, ch_in: int, dim_z: int):
        super().__init__()
        self.ch_in: int = ch_in
        self.width: int = size
        assert self.width == 28
        self.dim_z = dim_z

        self.conv = nn.Sequential(
            torch.nn.Conv2d(self.ch_in, 32, 4, 1, 2),  # B, 32, 28, 28
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, 4, 2, 1),  # B, 32, 14, 14
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, 4, 2, 1),  # B, 64,  7, 7
        )
        self.compute_mu = nn.Linear(64 * 7 * 7, self.dim_z)
        self.compute_std = nn.Linear(64 * 7 * 7, self.dim_z)

    def forward(self, x: torch.Tensor) -> ZZ:  # this is right

        independent_dim = list(x.shape[:-3])  # this might includes: enumeration, n_boxes, batch_size
        dependent_dim = list(x.shape[-3:])  # this includes: ch, width, height
        # assert dependent_dim == [self.ch_raw_image, self.width, self.width]
        x1 = x.view([-1] + dependent_dim)  # flatten the independent dimensions
        x2 = self.conv(x1).view(-1, 64 * 7 * 7)  # flatten the dependent dimension
        mu = self.compute_mu(x2).view(independent_dim + [self.dim_z])
        std = F.softplus(self.compute_std(x2)).view(independent_dim + [self.dim_z])
        return ZZ(mu=mu, std=std + EPS_STD)



class SimpleVae(torch.nn.Module):

    def __init__(self, params: dict) -> None:
        super().__init__()

        # Instantiate all the modules
        self.decoder: DecoderConv = DecoderConv(size=params["architecture"]["width_input_image"],
                                                dim_z=params["architecture"]["dim_z"],
                                                ch_out=params["architecture"]["ch_input_image"])

        self.encoder: EncoderConv = EncoderConv(size=params["architecture"]["width_input_image"],
                                                ch_in=params["architecture"]["ch_input_image"],
                                                dim_z=params["architecture"]["dim_z"])

        # Raw image parameters
        self.sigma = torch.nn.Parameter(data=torch.tensor(params["GECO_loss"]["fg_std"])[..., None, None],  # add singleton for width, height
                                        requires_grad=False)

        self.geco_dict = params["GECO_loss"]
        self.geco_balance_factor = torch.nn.Parameter(data=torch.tensor(self.geco_dict["factor_balance_range"][1]),
                                                      requires_grad=True)

        # Put everything on the cude if cuda available
        if torch.cuda.is_available():
            self.cuda()

    @staticmethod
    def NLL_MSE(output: torch.tensor, target: torch.tensor, sigma: torch.tensor) -> torch.Tensor:
        return ((output - target) / sigma).pow(2)

    def compute_metrics(self,
                        imgs_in: torch.Tensor,
                        imgs_out: torch.Tensor,
                        kl: torch.Tensor):

        # We are better off using MeanSquareError metric
        mse = SimpleVae.NLL_MSE(output=imgs_out, target=imgs_in.detach(), sigma=self.sigma)
        mse_av = mse.mean()  # mean over batch_size, ch, w, h

        # 4. compute the KL for each image
        kl_av = torch.mean(kl)  # mean over batch_size and latend dimension of z
        assert mse_av.shape == kl_av.shape

        # Note that I clamp in_place
        with torch.no_grad():
            f_balance = self.geco_balance_factor.data.clamp_(min=min(self.geco_dict["factor_balance_range"]),
                                                             max=max(self.geco_dict["factor_balance_range"]))
            one_minus_f_balance = torch.ones_like(f_balance) - f_balance

        # 6. Loss_VAE
        loss_vae = f_balance * mse_av + one_minus_f_balance * kl_av

        # GECO BUSINESS
        if self.geco_dict["is_active"]:
            with torch.no_grad():
                # If nll_av > max(target) -> tmp3 > 0 -> delta_2 < 0 -> bad reconstruction -> increase f_balance
                # If nll_av < min(target) -> tmp4 > 0 -> delta_2 > 0 -> too good reconstruction -> decrease f_balance
                tmp3 = (mse_av - max(self.geco_dict["target_mse"])).clamp(min=0)
                tmp4 = (min(self.geco_dict["target_mse"]) - mse_av).clamp(min=0)
                delta_2 = (tmp4 - tmp3).requires_grad_(False).to(loss_vae.device)

            loss_2 = self.geco_balance_factor * delta_2
            loss_av = loss_vae + loss_2 - loss_2.detach()
        else:
            delta_2 = torch.tensor(0.0, dtype=loss_vae.dtype, device=loss_vae.device)
            loss_av = loss_vae

        # add everything you want as long as there is one loss
        return MetricMiniBatchSimple(loss=loss_av,
                                     mse=mse_av.detach(),
                                     kl_tot=kl_av.detach(),
                                     geco_balance=f_balance,
                                     delta_2=delta_2)

    # this is the generic function which has all the options unspecified
    def process_batch_imgs(self,
                           imgs_in: torch.tensor,
                           generate_synthetic_data: bool,
                           noisy_sampling: bool):
        """ It needs to return: metric (with a .loss member) and whatever else """

        # Checks
        assert len(imgs_in.shape) == 4
        # End of Checks #

        z_inferred: ZZ = self.encoder.forward(imgs_in)
        z: DIST = sample_and_kl_diagonal_normal(posterior_mu=z_inferred.mu,
                                                posterior_std=z_inferred.std,
                                                prior_mu=torch.zeros_like(z_inferred.mu),
                                                prior_std=torch.ones_like(z_inferred.std),
                                                noisy_sampling=noisy_sampling,
                                                sample_from_prior=generate_synthetic_data)

        imgs_rec = torch.sigmoid(self.decoder.forward(z.sample))

        metrics = self.compute_metrics(imgs_in=imgs_in,
                                       imgs_out=imgs_rec,
                                       kl=z.kl)

        return OutputSimple(metrics=metrics, z=z.sample, imgs=imgs_rec)

    def forward(self, imgs_in: torch.tensor):
        return self.process_batch_imgs(imgs_in=imgs_in,
                                       generate_synthetic_data=False,
                                       noisy_sampling=True)  # True if self.training else False,

    def generate(self, imgs_in: torch.tensor):
        with torch.no_grad():
            return self.process_batch_imgs(imgs_in=torch.zeros_like(imgs_in),
                                           generate_synthetic_data=True,
                                           noisy_sampling=True).imgs
