import torch
import collections
from utilities import sample_normal, kl_normal0_normal1
from encoders_decoders import *


CHECKPOINT = collections.namedtuple("checkpoint", "model optimizer history_dict epoch params_dict")
METRIC = collections.namedtuple("metric", "loss kl nll")
INFERENCE = collections.namedtuple("inference", "reconstruction")


def save_everything(path, model, optimizer, history_dict, epoch, params):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history_dict': history_dict,
                'param_dict': params}, path)


def load_everything(path, model=None, optimizer=None):
    resumed = torch.load(path)

    params = resumed['params_dict']
    epoch = resumed['epoch']
    history_dict = resumed['history_dict']

    if model is None:
        model = VaeClass(params)
    model.load_state_dict(resumed['model_state_dict'])
    model.train()

    if optimizer is None:
        optimizer = instantiate_optimizer(model, params)
        optimizer.load_state_dict(resumed['optimizer_state_dict'])

    return CHECKPOINT(model=model, optimizer=optimizer, history_dict=history_dict, epoch=epoch, params_dict=params)


def instantiate_optimizer(model, params):
    if params["optimizer"]["type"] == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=params["optimizer"]["lr"],
                                     betas=params["optimizer"]["betas"],
                                     eps=params["optimizer"]["eps"])
    else:
        raise Exception
    return optimizer


class VaeClass(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.encoder = EncoderConv(params,
                                   ch_in=params["architecture"]["ch_input_image"],
                                   dim_z=params["architecture"]["dim_zwhat"])
        self.decoder = DecoderConv(params,
                                   dim_z=params["architecture"]["dim_zwhat"],
                                   ch_out=params["architecture"]["ch_input_image"])

        self.ch_input_image = params["architecture"]["ch_input_image"]
        self.width_input_image = params["architecture"]["width_input_image"]
        self.loss_dict = params["loss"]

        # Put everything on the cude if necessary
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def NLL_MSE(self, output=None, target=None, sigma=None):
        return ((output-target)/sigma).pow(2)

    def forward(self, imgs_in=None):

        # Trick #
        if imgs_in is None:
            observed = False
            imgs_in = torch.zeros(8, self.ch_input_image, self.width_input_image, self.width_input_image)
            if self.use_cuda and (imgs_in.device == "cpu"):
                imgs_in = imgs_in.cuda()
        else:
            observed = True
        assert len(imgs_in.shape) == 4
        batch_size, ch, width, height = imgs_in.shape
        assert width == height
        # End of Trick #

        zero = torch.zeros([1], dtype=imgs_in.dtype, device=imgs_in.device)
        one = torch.ones([1], dtype=imgs_in.dtype, device=imgs_in.device)

        zwhat = self.encoder.forward(imgs_in)
        zwhat_sample = sample_normal(mu=zwhat.mu, std=zwhat.std, noisy_sampling=True)
        kl_zwhat = kl_normal0_normal1(mu0=zwhat.mu, std0=zwhat.std, mu1=zero, std1=one).mean(dim=-1)  # mean over latent
        imgs_rec = self.decoder.forward(zwhat_sample)

        nll = self.NLL_MSE(output=imgs_rec,
                           target=imgs_in.detach(),
                           sigma=self.loss_dict["mse_sigma"]).mean(dim=(-1, -2, -3))  # mean over ch,w,h

        # assert kl_zwhat.shape == nll.shape

        loss = torch.sum(nll + kl_zwhat)

        return METRIC(loss=loss, kl=kl_zwhat, nll=nll), INFERENCE(reconstruction=imgs_rec)

    def reconstruct_img(self, imgs_in=None):

        if self.use_cuda and (imgs_in.device == "cpu"):
            imgs_in = imgs_in.cuda()

        with torch.no_grad():
            return self.forward(imgs_in)
