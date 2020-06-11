import torch
import torch.nn.functional as F
from torch import Tensor


def distance(x1: Tensor, x2: Tensor) -> Tensor:
    """
        As per TripletGan Code: https://github.com/maciejzieba/tripletGAN/blob/master/tgan_mnist.py#L75
        :returns Euclidean distance between two vector
    """
    return torch.sqrt(torch.sum(torch.pow(x1 - x2, 2), dim=1))


def f_discriminator_unsupervised_loss(fake: Tensor, real: Tensor) -> Tensor:
    """
    Unsupervised part the discriminator loss: L_T_u

    As per TripletGan Code: https://github.com/maciejzieba/tripletGAN/blob/3afa79161f6a02a9e3c98a714cca80e853d05384/tgan_mnist.py#L87
    :param fake: D(fake)
    :param real: D(real)
    :return:
    """
    realz = -0.5 * torch.mean(torch.logsumexp(real, dim=1)) + 0.5 * torch.mean(F.softplus(torch.logsumexp(real, dim=1)))
    fakez = 0.5 * torch.mean(F.softplus(torch.logsumexp(fake, dim=1)))
    return realz + fakez


def triplet_paper_loss(anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
    """
    Triplet loss as given in the paper

    As per TripletGan Code: https://github.com/maciejzieba/tripletGAN/blob/master/tgan_mnist.py#L74-L84
    :param anchor: D(anchor)
    :param positive: D(positive)
    :param negative: D(negative)
    :return:
    """

    distance_positive = torch.exp(distance(anchor, positive))
    distance_negative = torch.exp(distance(anchor, negative))

    distance_cute_cat = torch.cat([distance_positive, distance_negative], dim=1)
    distance_cute_cat = torch.logsumexp(distance_cute_cat, dim=1)

    # https://github.com/maciejzieba/tripletGAN/blob/3afa79161f6a02a9e3c98a714cca80e853d05384/tgan_mnist.py#L84
    loss = - torch.mean(distance_negative) + torch.mean(distance_cute_cat)

    return loss


def feature_matching_loss(input: Tensor, target: Tensor) -> Tensor:
    """
        feature matching loss!!!
        As per TripletGan Code: https://github.com/maciejzieba/tripletGAN/blob/master/tgan_mnist.py#L90-L92

        TOOO: Why not MSELoss? Why are we taking this mean like: torch.mean(input, dim=0)?
    """
    fake = torch.mean(input, dim=0)
    real = torch.mean(target, dim=0)
    return torch.mean(torch.pow((fake - real), 2))
