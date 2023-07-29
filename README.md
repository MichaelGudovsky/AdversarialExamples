# AdversarialExamples
# CCUAP-Conditional Cumulative Universal Perturbation
This project aims to generalize the Deep-Fool algorithm 

The dataset is taken from the ImageNet ILSVRC2012 challenge and can be downloaded from the following link:
https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
Download the data and move the training data and test data to the project folder separately
The project was designed to run on the Nvidia GPU accelerator on Ubuntu 21.04 operating system

To run the CCUAP model, run the perturbate_image.py located in the proposed_method folder
The pre-trained models are stored in the folder models_part

To evaluate the perturbation run evaluate_model.py 

To view the comparison based on SNR and SSIM run snr.py and SSIM.py

In order to see the results of a simple averaging replace in the file perturbate image for checking DF.npy instead proposed.npy replacing line 20 with line 21
additionally, line 29 should replace line 28.
choose the wanted perturbation method by changing methods in lines 52-63

With the pre-trained models' existence and a small test set, we have a demo run to see the perturbation and evaluate it.
Because of the small test set' the accuracy is not as reported in the research

In order to see the images from different models and SNR look in the folder result_images.

The folder saliency map is added as part of our research for universal perturbation

CleverHans is python adversarial example package 


# DeepFool
DeepFool is a simple algorithm to find the minimum adversarial perturbations in deep networks

### deepfool.py

This function implements the algorithm proposed in [[1]](http://arxiv.org/pdf/1511.04599) using PyTorch to find adversarial perturbations.

__Note__: The final softmax (loss) layer should be removed in order to prevent numerical instabilities.

The parameters of the function are:

- `image`: Image of size `HxWx3d`
- `net`: neural network (input: images, output: values of activation **BEFORE** softmax).
- `num_classes`: limits the number of classes to test against, by default = 10.
- `max_iter`: max number of iterations, by default = 50.

### test_deepfool.py

A simple demo which computes the adversarial perturbation for a test image from ImageNet dataset.

## Reference
[1] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard:
*DeepFool: a simple and accurate method to fool deep neural networks*.  In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.

