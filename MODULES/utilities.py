import torch
import numpy
from torchvision import utils
from matplotlib import pyplot as plt
from typing import Union, Optional, Callable, Tuple
from .namedtuple import DIST
from collections import OrderedDict
import json
from torch.distributions.utils import broadcast_all


def sample_and_kl_diagonal_normal(posterior_mu: torch.Tensor,
                                  posterior_std: torch.Tensor,
                                  prior_mu: torch.Tensor,
                                  prior_std: torch.Tensor,
                                  noisy_sampling: bool,
                                  sample_from_prior: bool) -> DIST:

    post_mu, post_std, pr_mu, pr_std = broadcast_all(posterior_mu, posterior_std, prior_mu, prior_std)
    if sample_from_prior:
        # working with the prior
        sample = pr_mu + pr_std * torch.randn_like(pr_mu) if noisy_sampling else pr_mu
        kl = torch.zeros_like(pr_mu)
    else:
        # working with the posterior
        sample = post_mu + post_std * torch.randn_like(post_mu) if noisy_sampling else post_mu
        tmp = (post_std + pr_std) * (post_std - pr_std) + (post_mu - pr_mu).pow(2)
        kl = tmp / (2 * pr_std * pr_std) - post_std.log() + pr_std.log()

    return DIST(sample=sample, kl=kl)


def flatten_list(ll):
    if not ll:  # equivalent to if ll == []
        return ll
    elif isinstance(ll[0], list):
        return flatten_list(ll[0]) + flatten_list(ll[1:])
    else:
        return ll[:1] + flatten_list(ll[1:])


def flatten_dict(dd, separator='_', prefix=''):
    return {prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
            } if isinstance(dd, dict) else {prefix: dd}


def save_obj(obj, path):
    with open(path, 'wb') as f:
        torch.save(obj, f, 
                pickle_module=<module 'pickle' from '/opt/conda/lib/python3.6/pickle.py'>,
                pickle_protocol=2, 
                _use_new_zipfile_serialization=True)


def load_obj(path):
    with open(path, 'rb') as f:
        return torch.load(f, 
                pickle_module=<module 'pickle' from '/opt/conda/lib/python3.6/pickle.py'>,
                pickle_protocol=2, 
                _use_new_zipfile_serialization=True)


def load_json_as_dict(path):
    with open(path, 'rb') as f:
        return json.load(f)


def save_dict_as_json(my_dict, path):
    with open(path, 'w') as f:
        return json.dump(my_dict, f)


class Accumulator(object):
    """ accumulate a tuple or tuple into a dictionary.
        At the end returns the tuple with the average values """

    def __init__(self):
        super().__init__()
        self._counter = 0
        self._dict_accumulate = OrderedDict()
        self._input_cls = None

    def accumulate(self, input: Union[tuple, dict], counter_increment: int = 1):

        if self._input_cls is None:
            self._input_cls = input.__class__
        else:
            assert isinstance(input, self._input_cls)

        if isinstance(input, tuple):
            input_dict = input._asdict()
        elif isinstance(input, dict) or isinstance(input, OrderedDict):
            input_dict = input
        else:
            raise Exception

        self._counter += counter_increment

        for k, v in input_dict.items():
            try:
                self._dict_accumulate[k] = v * counter_increment + self._dict_accumulate[k]
            except KeyError:
                self._dict_accumulate[k] = v * counter_increment

    def get_cumulative(self):
        return self.export(self._dict_accumulate)

    def get_average(self):
        tmp = self._dict_accumulate
        for k, v in self._dict_accumulate.items():
            tmp[k] = v/self._counter
        return self.export(tmp)

    def export(self, od):
        if self._input_cls == dict:
            return dict(od)
        elif self._input_cls == OrderedDict:
            return od
        else:
            return self._input_cls._make(od.values())


class SpecialDataSet(object):
    def __init__(self,
                 img: torch.Tensor,
                 roi_mask: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None,
                 data_augmentation: Optional[Callable] = None,
                 store_in_cuda: bool = False,
                 drop_last=False,
                 batch_size=4,
                 shuffle=False):
        """ :param device: 'cpu' or 'cuda:0'
            Dataset returns random crops of a given size inside the Region Of Interest.
            The function getitem returns imgs, labels and indeces
        """
        assert len(img.shape) == 4
        assert (roi_mask is None or len(roi_mask.shape) == 4)
        assert (labels is None or labels.shape[0] == img.shape[0])

        storing_device = torch.device('cuda') if store_in_cuda else torch.device('cpu')

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Expand the dataset so that I can do one crop per image
        if data_augmentation is None:
            new_batch_size = img.shape[0]
            self.data_augmentaion = None
        else:
            new_batch_size = img.shape[0] * data_augmentation.n_crops_per_image
            self.data_augmentaion = data_augmentation

        if store_in_cuda:
            self.img = img.cuda().detach().expand(new_batch_size, -1, -1, -1)
        else:
            self.img = img.cpu().detach().expand(new_batch_size, -1, -1, -1)

        if labels is None:
            self.labels = -1*torch.ones(self.img.shape[0], device=storing_device).detach()
        else:
            self.labels = labels.to(storing_device).detach()
        self.labels = self.labels.expand(new_batch_size)

        if roi_mask is None:
            self.roi_mask = None
            self.cum_roi_mask = None
        else:
            self.roi_mask = roi_mask.to(storing_device).detach().expand(new_batch_size, -1, -1, -1)
            self.cum_roi_mask = roi_mask.to(storing_device).detach().cumsum(dim=-1).cumsum(dim=-2).expand(new_batch_size, -1, -1, -1)

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index: torch.Tensor):
        assert isinstance(index, torch.Tensor)

        if self.data_augmentaion is None:
            return self.img[index], self.labels[index], index
        else:
            bij_list = []
            for i in index:
                bij_list += self.data_augmentaion.get_index(img=self.img[i],
                                                            cum_sum_roi_mask=self.cum_roi_mask[i],
                                                            n_crops_per_image=1)
            return self.data_augmentaion.collate_crops_from_list(self.img, bij_list), self.labels[index], index

    def __iter__(self, batch_size=None, drop_last=None, shuffle=None):
        # If not specified use defaults
        batch_size = self.batch_size if batch_size is None else batch_size
        drop_last = self.drop_last if drop_last is None else drop_last
        shuffle = self.shuffle if shuffle is None else shuffle

        # Actual generation of iterator
        n_max = max(1, self.__len__() - (self.__len__() % batch_size) if drop_last else self.__len__())
        index = torch.randperm(self.__len__()).long() if shuffle else torch.arange(self.__len__()).long()
        for pos in range(0, n_max, batch_size):
            yield self.__getitem__(index[pos:pos + batch_size])
            
    def load(self, batch_size=None, index=None):
        if (batch_size is None and index is None) or (batch_size is not None and index is not None):
            raise Exception("Only one between batch_size and index must be specified")
        index = torch.randint(low=0, high=self.__len__(), size=(batch_size,)).long() if index is None else index
        return self.__getitem__(index)

    def check_batch(self, batch_size: int = 8):
        print("Dataset lenght:", self.__len__())
        print("img.shape", self.img.shape)
        print("img.dtype", self.img.dtype)
        print("img.device", self.img.device)
        index = torch.randperm(self.__len__(), dtype=torch.long, device=self.img.device, requires_grad=False)
        # grab one minibatch
        img, labels, index = self.__getitem__(index[:batch_size])
        print("MINIBATCH: img.shapes labels.shape, index.shape ->", img.shape, labels.shape, index.shape)
        print("MINIBATCH: min and max of minibatch", torch.min(img), torch.max(img))
        return show_batch(img, n_col=4, n_padding=4, pad_value=1, figsize=(24, 24))


def process_one_epoch(model: torch.nn.Module,
                      dataloader: SpecialDataSet,
                      optimizer: Optional[torch.optim.Optimizer] = None,
                      weight_clipper: Optional[Callable[[None], None]] = None,
                      verbose: bool = False) -> dict:
    """ return a tuple with all the metrics averaged over a epoch """
    metric_accumulator = Accumulator()

    for i, data in enumerate(dataloader):
        imgs, labels, index = data
        
        # Put data in GPU if available
        if torch.cuda.is_available() and imgs.device == torch.device('cpu'):
            imgs = imgs.cuda()
            labels = labels.cuda()
            index = index.cuda()
            
        metrics = model.forward(imgs_in=imgs).metrics  # the forward function returns metric and other stuff
        if verbose:
            print("i = %3d train_loss=%.5f" % (i, metrics.loss))

        # Accumulate metrics over an epoch
        with torch.no_grad():
            metric_accumulator.accumulate(input=metrics, counter_increment=len(index))

        # Only if training I apply backward
        if model.training:
            optimizer.zero_grad()
            metrics.loss.backward()  # do back_prop and compute all the gradients
            optimizer.step()  # update the parameters
        
            # apply the weight clipper
            if weight_clipper is not None:
                model.__self__.apply(weight_clipper)
                
        # Delete stuff from GPU
        # del imgs
        # del labels
        # del index
        # del metrics
        # torch.cuda.empty_cache()

    # At the end of the loop compute the average of the metrics
    with torch.no_grad():
        return metric_accumulator.get_average()


def show_batch(images: torch.Tensor,
               n_col: int = 4,
               n_padding: int = 10,
               title: Optional[str] = None,
               pad_value: int = 1,
               normalize_range: Optional[tuple] = None,
               figsize: Optional[Tuple[float, float]] = None):
    """Visualize a torch tensor of shape: (batch x ch x width x height) """
    assert len(images.shape) == 4  # batch, ch, width, height
    if images.device != "cpu":
        images = images.cpu()
    if normalize_range is None:
        grid = utils.make_grid(images, n_col, n_padding, normalize=False, pad_value=pad_value)
    else:
        grid = utils.make_grid(images, n_col, n_padding, normalize=True, range=normalize_range,
                               scale_each=False, pad_value=pad_value)
        
    fig = plt.figure(figsize=figsize)
    plt.imshow(grid.detach().permute(1,2,0).squeeze(-1).numpy())
    if isinstance(title, str):
        plt.title(title)
    plt.close(fig)
    fig.tight_layout()
    return fig
