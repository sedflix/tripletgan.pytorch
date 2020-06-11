from random import random

import torchvision.datasets as dataset_
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from .dataset import TripletMNISTDataset
from .losses import *
from .model import *
from .utills import *

image_size = 28
z_feautures = 100
num_epochs = 50
lr = 0.0002
beta1 = 0.5
batch_size = 64
out_dimension = 16

exp_name = "testing"

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

"""
    DATASET AND DATALOADER CREATION
"""
# setup dataset
normal_mnist_dataset = dataset_.MNIST(root="./data", train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize((.5,), (.5,))
                                      ]))
triplet_mnist_dataset = TripletMNISTDataset(normal_mnist_dataset, num_samples=60000)


# setup loader
normal_mnist_dataloader = data.DataLoader(normal_mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
triplet_mnist_dataloader = data.DataLoader(triplet_mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


"""
    MODEL CREATION
"""

# generator and  discriminator
generator_net = Generator(z_feautures).to(device)
discriminator_net = Discriminator(out_dimension).to(device)

# create optimiser for the same
discriminator_optimizer = Adam(discriminator_net.parameters(), lr=lr, betas=(beta1, 0.999))
generator_optimizer = Adam(generator_net.parameters(), lr=lr, betas=(beta1, 0.999))

# fixed normal noise that is used to generate fake images after sometimes
fixed_noise = torch.randn((32, z_feautures, 1, 1), device=device)

writer = SummaryWriter(f"./data/run/{exp_name}")


# images for visualising embeddings made by the discriminator
special_batch = {
    'imgs': [],
    'labels': [],
    'features': [],
}
per_class_sample = 10
for label in triplet_mnist_dataset.classes:
    special_batch['imgs'].extend(random.choices(triplet_mnist_dataset.labels2images[label], k=per_class_sample))
    special_batch['labels'].extend([label for _ in range(per_class_sample)])
special_batch['imgs'] = torch.cat(special_batch['imgs']).reshape(-1, 1, image_size, image_size)


"""
    TRAINING GOING TO START
"""
visualisation_imgs = []  # stores visulization of model at the end of each x steps
generator_losses = []
discriminator_losses = []
steps = 1

for epoch in range(num_epochs):
    for i, (anchor_img, positive_img, negative_img) in enumerate(triplet_mnist_dataloader):
        real_imgs, _ = next(iter(normal_mnist_dataloader))
        real_imgs.to(device)
        anchor_img, positive_img, negative_img = anchor_img.to(device), positive_img.to(device), negative_img.to(device)

        """
            Partially Supervised Discriminator Loss:
            Update discriminator net using triplet loss!
        """
        discriminator_net.zero_grad()
        anchor_out, positive_out, negative_out = discriminator_net(anchor_img), discriminator_net(
            positive_img), discriminator_net(negative_img)
        discriminator_loss_triplet = triplet_paper_loss(anchor_out, positive_out, negative_out)
        discriminator_loss_triplet.backward()
        discriminator_optimizer.step()  # <- Update discriminator

        """
            Unsupervised Discriminator Loss:
            Update discriminator net using fake and real data
        """
        discriminator_net.zero_grad()
        fake_imgs = generator_net(torch.randn(real_imgs.size(0), z_feautures, 1, 1, device=device))
        real_output = discriminator_net(real_imgs.detach())
        fake_output = discriminator_net(fake_imgs.detach())
        discriminator_loss_unsupervised = f_discriminator_unsupervised_loss(fake_output, real_output)
        discriminator_loss_unsupervised.backward()
        discriminator_optimizer.step()  # <- Update discriminator

        discriminator_loss_total = discriminator_loss_triplet + discriminator_loss_unsupervised

        """
            Update out generator : Done!!
        """
        generator_net.zero_grad()
        discriminator_net.zero_grad()

        real_output = discriminator_net(real_imgs)
        fake_output = discriminator_net(fake_imgs)

        # feature_matching_loss has been used as generator_loss
        generator_loss = feature_matching_loss(fake_output, real_output)
        generator_loss.backward()

        generator_optimizer.step()  # <- Update  generator

        generator_losses.append(generator_loss.item())
        discriminator_losses.append(discriminator_loss_total.item())

        writer.add_scalars('Loss', {
            'discriminator': discriminator_loss_total.item(),
            'generator': generator_loss.item()
        }, steps)
        steps = steps + 1

        print(
            f"Epoch: {epoch:3d}; Iteration: {i:4d}; "
            f"Loss D: {discriminator_loss_total.item():0.4f}; "
            f"Loss G: {generator_loss.item():0.4f}; ",
            end="\r"
        )

    # make images from fixed noise for visualization
    with torch.no_grad():
        fake_images_with_fixed_noise = generator_net(fixed_noise)
    grid = utils.make_grid(fake_images_with_fixed_noise.detach().cpu(), padding=2, normalize=True)
    visualisation_imgs.append(grid)

    # add image to tb
    writer.add_image("generated image", grid, steps)

    # write model graphs to tb
    # writer.add_graph(generator_net, fixed_noise)
    # writer.add_graph(discriminator_net, fake_images_with_fixed_noise)

    # add embedding of b to tb
    with torch.no_grad():
        special_batch['features'] = discriminator_net(special_batch['imgs'].to(device)).detach().cpu()
    writer.add_embedding(special_batch['features'].reshape(-1, out_dimension),
                         global_step=steps,
                         metadata=special_batch['labels'],
                         label_img=special_batch['imgs'])

    checkpoint(generator_net, "generator", exp_name)
    checkpoint(discriminator_net, "discriminator", exp_name)


writer.close()
plot_losses(generator_losses, discriminator_losses)
plot_animation(visualisation_imgs, exp_name)
plt.imshow()
