from sklearn import preprocessing
import torch
import logging
from omegaconf import DictConfig

# import utils.logger as logger
# from utils.read_yaml import config

log = logging.getLogger("utils.transform")
# log = logger.get_logger("utils.transform")


def to_physical(x, param, cfg: DictConfig):
    """
    The reverse scaling function to find the physical param from the scaled param (range [0,1))
    x: the value, or array, you want to turn from a random number into a physical value
    param: the string of the variable you want to transform
    :return:
    """
    range_ = cfg.transformation[param]
    x_ = x * (range_[1] - range_[0])
    output = x_ + range_[0]
    return output


def from_physical(x, param):
    """
    The scaling function to convert a physical param to a value within [0,1)
    x: the value, or array, you want to turn from a random number into a physical value
    param: the string of the variable you want to transform
    :return:
    """


def normalization(x, min_x, max_x):
    """
    A min/max Scaler for each feature to be fed into the MLP
    :param x:
    :return:
    """
    normalized_x = (x - min_x) / (max_x - min_x)

    return normalized_x
