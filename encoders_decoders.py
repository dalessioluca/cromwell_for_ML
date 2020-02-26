import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

ZZ = collections.namedtuple("encoded", "mu std")


class DecoderConv(nn.Module):
    """ Decode z -> x
        INPUT:  z of shape: ..., dim_z 
        OUTPUT: image of shape: ..., ch_out, width, height 
        where ... are all the independent dimensions, i.e. box, batch_size, enumeration_dim etc.
    """
    def __init__(self, params, dim_z: int, ch_out: int):
        super().__init__()
        self.width = params["architecture"]["width_input_image"]
        assert self.width == 28
        self.dim_z = dim_z
        self.ch_out = ch_out
        self.upsample = nn.Linear(self.dim_z, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, self.ch_out, 4, 1, 2)  # B, ch, 28, 28
        )

    def forward(self, z):
        independent_dim = list(z.shape[:-1])
        x1 = self.upsample(z.view(-1, self.dim_z)).view(-1, 64, 7, 7)
        return self.decoder(x1).view(independent_dim + [self.ch_out, self.width, self.width])


class EncoderConv(nn.Module):
    """ Encode x -> z_mu, z_std
        INPUT  x of shape: ..., ch_raw_image, width, height 
        OUTPUT z_mu, z_std of shape: ..., latent_dim
        where ... are all the independent dimensions, i.e. box, batch_size, enumeration_dim etc.
    """ 
    
    def __init__(self, params, ch_in: int, dim_z: int):
        super().__init__()
        self.ch_in = ch_in
        self.width = params["architecture"]["width_input_image"]
        assert self.width == 28
        self.dim_z = dim_z

        self.conv = nn.Sequential(
            torch.nn.Conv2d(self.ch_in, 32, 4, 1, 2),  # B, 32, 28, 28
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, 2, 1),  # B, 32, 14, 14
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2, 1),  # B, 64,  7, 7
        )
        self.compute_mu = nn.Linear(64 * 7 * 7, self.dim_z)
        self.compute_std = nn.Linear(64 * 7 * 7, self.dim_z)

    def forward(self, x):  # this is right

        independent_dim = list(x.shape[:-3])  # this might includes: enumeration, n_boxes, batch_size
        dependent_dim = list(x.shape[-3:])  # this includes: ch, width, height
        # assert dependent_dim == [self.ch_raw_image, self.width, self.width]
        x1 = x.view([-1] + dependent_dim)  # flatten the independent dimensions
        x2 = self.conv(x1).view(-1, 64*7*7)  # flatten the dependent dimension
        mu = self.compute_mu(x2).view(independent_dim + [self.dim_z])
        std = F.softplus(self.compute_std(x2)).view(independent_dim + [self.dim_z]) + 1E-3
        return ZZ(mu=mu, std=std)
