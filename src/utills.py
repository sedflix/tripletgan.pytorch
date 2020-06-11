from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML
from matplotlib import animation
from torch import Tensor, rand
from torchvision import utils


def checkpoint(model: torch.nn.Module, name: str, exp_name: str):
    torch.save(model, f"./data/{name}_{exp_name}.model")
    torch.save(model.state_dict(), f"./data/{name}_{exp_name}.state_dict")


def plot_losses(generator_losses: List[float], discriminator_losses: List[float]):
    plt.figure(figsize=(12, 6))
    plt.plot(generator_losses, label="generator loss")
    plt.plot(discriminator_losses, label="discriminator_ loss")
    plt.ylabel("loss")
    plt.xlabel("num_steps")
    plt.legend()


def plot_animation(visualisation_imgs: List[Any], exp_name):
    fig = plt.figure(figsize=(24, 24))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in visualisation_imgs]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    Writer = animation.writers['ffmpeg']
    ani.save(f"./data/{exp_name}_step.mp4", Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800))
    HTML(ani.to_jshtml())


def plot_image_grid(batched_image, save_path=None):
    image_grid = utils.make_grid(batched_image, padding=2, normalize=True)
    image_grid = np.transpose(image_grid, (1, 2, 0))
    plt.imshow(image_grid)

    if save_path is not None:
        plt.imsave(save_path, image_grid)


def get_real_label(b: int) -> Tensor:
    """
    create labels for a real batch in discriminator
    it is randomly between 1.2 and 0.8
    :param b: batch size
    """
    return rand(b) * (1.2 - 0.8) + 1.0


def get_fake_label(b: int) -> Tensor:
    """
     create labels for a fake batch in discriminator.
     it is randomly between 0.3 and 0
    :param b: batch size
    """
    return rand(b) * (0.3 - 0.0) + 0.0
