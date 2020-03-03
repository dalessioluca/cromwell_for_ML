import torch
import pickle
import json
from torchvision import utils
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset
from torch.distributions.utils import broadcast_all


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_json_as_dict(path):
    with open(path, 'rb') as f:
        return json.load(f)


def save_dict_as_json(my_dict, path):
    with open(path, 'w') as f:
        return json.dump(my_dict, f)


def reset_parameters(parent_module, verbose):
    for m in parent_module.modules():
        try:
            m.reset_parameters()
            if verbose:
                print("reset -> ", m)
        except AttributeError:
            pass


def add_named_tuple_to_dictionary(namedtuple, dictionary, key_prefix=None):
    with torch.no_grad():

        for k, v in namedtuple.items():

            new_k = k if key_prefix is None else key_prefix+k
            try:
                dictionary[new_k].append(v)
            except KeyError:
                dictionary[new_k] = [v]

        return dictionary


def sample_normal(mu, std, noisy_sampling):
    new_mu, new_std = broadcast_all(mu, std)
    if noisy_sampling:
        return new_mu + new_std * torch.randn_like(new_mu)
    else:
        return new_mu


def kl_normal0_normal1(mu0, mu1, std0, std1):
    tmp = (std0 + std1) * (std0 - std1) + (mu0 - mu1).pow(2)
    return tmp / (2 * std1 * std1) - torch.log(std0 / std1)


def train_one_epoch(model, dataset, optimizer, batch_size=64, verbose=False, weight_clipper=None):
    """ return the average of the metric over one epoch"""
    n_term = 0
    batch_iterator = dataset.generate_batch_iterator(batch_size)
    metric_av = {}

    for i, indices in enumerate(batch_iterator):  # get the indeces

        # get the data from the indices. Note that everything is already loaded into memory
        metric, inference = model.forward(imgs_in=dataset[indices][0])
        if verbose:
            print("i = %3d train_loss=%.5f" % (i, metric.loss))

        # For each minibatch set the gradient to zero
        optimizer.zero_grad()
        metric.loss.backward()  # do backprop and compute all the gradients
        optimizer.step()  # update the parameters

        # apply the weight clipper
        if weight_clipper is not None:
            model.__self__.apply(weight_clipper)

        # Compute the average metric over a epoch
        with torch.no_grad():
            n_term += len(indices)
            for key in metric._fields:
                value = getattr(metric, key).sum().item()
                metric_av[key] = value + metric_av.get(key, 0.0)

    for k, v in metric_av.items():
        metric_av[k] = v/n_term
    return metric_av


def evaluate_one_epoch(model, dataset, batch_size=64, verbose=False):
    """ return the average of the metric over one epoch"""
    n_term = 0
    metric_av = {}
    batch_iterator = dataset.generate_batch_iterator(batch_size)

    with torch.no_grad():
        for i, indices in enumerate(batch_iterator):  # get the indeces

            # get the data from the indices. Note that everything is already loaded into memory
            metric, _ = model.forward(imgs_in=dataset[indices][0])
            if verbose:
                print("i = %3d train_loss=%.5f" % (i, metric.loss))

            # Compute the average metric over a epoch
            n_term += len(indices)
            for key in metric._fields:
                value = getattr(metric, key).sum().item()
                metric_av[key] = value + metric_av.get(key, 0.0)

        for k, v in metric_av.items():
            metric_av[k] = v / n_term
        return metric_av


def show_batch(images, n_col=4, n_padding=10, title=None, pad_value=1):
    """Visualize a torch tensor of shape: (batch x ch x width x height) """
    assert len(images.shape) == 4  # batch, ch, width, height
    if images.device != "cpu":
        images = images.cpu()
    grid = utils.make_grid(images, n_col, n_padding, normalize=True, range=(0.0, 1.0),
                           scale_each=False, pad_value=pad_value)
        
    fig = plt.figure()
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))
    if isinstance(title, str):
        plt.title(title)
    plt.close(fig)
    fig.tight_layout()
    return fig


class DatasetInMemory(Dataset):
    """ Typical usage:

        synthetic_data_test  = dataset_in_memory(root_dir,"synthetic_data_DISK_test_v1",use_cuda=False)
        for epoch in range(5):
            print("EPOCH")
            batch_iterator = synthetic_data_test.generate_batch_iterator(batch_size)
            for i, x in enumerate(batch_iterator):
                print(x)
                blablabla
                ......
    """

    def __init__(self, path, use_cuda=False):
        self.use_cuda = use_cuda

        with torch.no_grad():
            # this is when I have both imgs and labels
            imgs, labels = load_obj(path)
            self.n = imgs.shape[0]

            if self.use_cuda:
                self.data = imgs.cuda()
                self.labels = labels.cuda()
            else:
                self.data = imgs.cpu().detach()
                self.labels = labels.cpu().detach()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.data[index, ...], self.labels[index]

    def load(self, batch_size=8, indices=None):
        if indices is None:
            indices = torch.randint(low=0, high=self.n, size=(batch_size,)).long()
        return self.__getitem__(indices[:batch_size])

    def generate_batch_iterator(self, batch_size):
        indices = torch.randperm(self.n).numpy()
        remainder = len(indices) % batch_size
        n_max = len(indices) - remainder  # Note the trick so that all the minibatches have the same size
        n_max = len(indices)
        assert (n_max >= batch_size), "batch_size is too big for this dataset of size."
        batch_iterator = (indices[pos:pos + batch_size] for pos in range(0, n_max, batch_size))
        return batch_iterator

    def check(self):

        imgs, labels = self.load(batch_size=8)
        title = "# labels =" + str(labels.cpu().numpy().tolist())
        
        print("Dataset lenght:", self.__len__())
        print("imgs.shape", imgs.shape)
        print("type(imgs)", type(imgs))
        print("imgs.device", imgs.device)
        print("torch.max(imgs)", torch.max(imgs))
        print("torch.min(imgs)", torch.min(imgs))
        
        return show_batch(imgs, n_col=4, n_padding=4, title=title, pad_value=1)