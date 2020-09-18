import neptune
import torch
from typing import Union
from collections import OrderedDict


def log_model_summary(experiment: neptune.experiments.Experiment, model: torch.nn.Module):
    for x in model.__str__().split('\n'):
        # replace leading spaces with '-' character
        n = len(x) - len(x.lstrip(' '))
        experiment.log_text("model summary", '-'*n + x)


def log_metrics(experiment: neptune.experiments.Experiment, metrics: Union[tuple, dict], prefix: str = ""):

    if isinstance(metrics, tuple):
        input_dict = metrics._asdict()
    elif isinstance(metrics, dict) or isinstance(metrics, OrderedDict):
        input_dict = metrics
    else:
        print(type(metrics))
        Exception("metrics have a not recognized type")

    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            experiment.log_metric(prefix+key, value.item())
        elif isinstance(value, float):
            experiment.log_metric(prefix+key, value)


def log_last_ckpt(experiment: neptune.experiments.Experiment, path: str):
    experiment.delete_artifacts(path)
    experiment.log_artifact(path)
