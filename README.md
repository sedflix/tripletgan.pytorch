# TripletGAn

## Setup

- **pytorch==1.5.0**
- **torchvision==0.6.0**
- matplotlib
- scikit-learn 

## Usage

* notebooks/tain_triplet_gan_experiment.ipynb: a note that contains the a working training code for TripletGan and its evaluation using K-NN
* src/dataset.py: on how to create triplet dataset
* src/losses.py: various loss functions like Triplet Loss(as per in the paper), Unsupervised Discriminator loss, Feature Matching loss
* src/model.py: contains the model code for Generator and Discriminator. 
* src/train.py: contains a template code for training a triplet gan and loading dataset
* src/data: contains various checkpoints of the model

## Some results

Each model has been trained for **50 epochs** where **each epoch had exactly 60,000 randomly selected triplets**

<table>
  <tr>
   <td>(9-NN for N=100)
   </td>
   <td>m=16
   </td>
   <td>m=32
   </td>
  </tr>
  <tr>
   <td>Accuracy
   </td>
   <td>0.989
   </td>
   <td>0.9903
   </td>
  </tr>
  <tr>
   <td>mAP
   </td>
   <td>0.99151
   </td>
   <td>0.99253
   </td>
  </tr>
</table>


<table>
  <tr>
   <td>(9-NN for m=16)
   </td>
   <td>N=100
   </td>
   <td>N=200
   </td>
  </tr>
  <tr>
   <td>Accuracy
   </td>
   <td>0.9890
   </td>
   <td>0.9898
   </td>
  </tr>
  <tr>
   <td>mAP
   </td>
   <td>0.99151
   </td>
   <td> 0.99178
   </td>
  </tr>
</table>


The accuracy I found is considerably higher than what was reported in the paper. A possible reason for it might be that they trained there GAN only use 100 labeled examples from the training set whereas I used all of the training examples.

But the good thing is the accuracy follows the intuition of increasing when we go from m=16 to m=32 and from N=100 to N=200.


## Things tried

Even after trying a lot of hyperparameter tuning and ways of training a vanilla GAN(with linear layers), I wasn't able to get the model to train properly. What a tried with Vanilla GAN:

- adding normal noise to the input of discriminator 
- making positive labels between 1.2 and 0.8 and negative labels between 0.0 and 0.3
- BatchNorm
- Dropout
- LeakyReLU
- weight normalization of the last layer of the generator and all layers of the discriminator 
- Normalization of image inputs
- Increasing and decreasing number of units in linear layers
- SGD for Discriminator and different learning rate

Hence, I shifted to** DCGAN! **

With DCGAN, I implemented all of the above things by default and I tried the following loss functions while training the TripletFAN

- Triplet Margin Loss + HingeLoss + Hinge Loss: didn't work
- Triplet Margin Loss + HingeLoss + feature matching loss: didn't work
- Triplet Margin Loss + f_discriminator_unsupervised_loss + feature matching: worked nicely
- Triplet Loss as in paper  + f_discriminator_unsupervised_loss + feature matching loss: worked nicely 


## References: 

- For DCGAN (the basic template of this code was taken from here)
    - https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- For various hacks for gan models and trainings:
    - https://github.com/soumith/ganhacks/
    - https://github.com/Sleepychord/ImprovedGAN-pytorch
- For details related to the paper:
    - https://github.com/maciejzieba/tripletGAN
    - https://arxiv.org/pdf/1704.02227.pdf
