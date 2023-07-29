# AdversarialExamples

This project aims to generalize the Deep-Fool algorithm (13.	Moosavi-Dezfooli, S. M., Fawzi, A., & Frossard, P. (2016). 
Deepfool: a simple and accurate method to fool deep neural networks) to be image and network agnostic

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

With the pre-trained models' existence and a small test set, we have a demo run to see the perturbation and evaluate it.
Because of the small test set' the accuracy is not as reported in the research

In order to see the images from different models and SNR look in the folder result_images.

The folder saliency map is added as part of our research for universal perturbation


