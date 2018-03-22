# MNIST/SVHN/CIFAR-10 experiments


The file added to the code of GoodFellow:

File: train_cifar_feature_matching_scale.py
This code is a modification of Goodfellow code for GAN
The generator is asked to create a CIFAR image rescaled, the new image has size of 2 or 4 times bigger then the original one 
It is possible to change the value of the parameter "scale" to rescale the size of cifar up to 4


File: train_cifar_feature_matching_scale_generator.py
This code is a modification of Goodfellow code for GAN
The generator receive an image in input and is asked to create an image to fake the discriminator. In order to fake the discriminator, the distance between the activation of the discriminator when seeing a real image I and the activation when seeing the output of the generator when the generator has as input I is used as error signal for the generator.
The code trains the discriminator and the generator on CIFAR, it is possible to change the value of the parameter "scale" to rescale the size of cifar up to 4

## OLD README
This part of the code is built using Theano and Lasagne. Any recent version of these packages should work for running the code.

The experiments are run using the train*.py files. All experiments perform semi-supervised learning with a set of labeled examples and a set of unlabeled examples. There are two kinds of models: the "feature matching" models that achieve the best predictive performance, and the "minibatch discrimination" models that achieve the best image quality.

The provided train*.py files each train a single model for a single random labeled/unlabeled data split and a single random parameter initialization. To reproduce our results using ensembling / averaging over random seeds, you can run these files multiple times using different inputs for the "seed" and "seed_data" arguments, and then combine the results.

This code is still being developed and subject to change.

