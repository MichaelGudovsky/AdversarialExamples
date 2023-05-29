import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.models as models
from evaluate_model import evaluate
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

clean_path = './test'
noisy_path = './noisy_images'
cropped_path = './cropped'



np.random.seed(42)
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device)
resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(device)
googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT).to(device)
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(device)
classifiers =  [alexnet, googlenet, resnet, vgg]

count = 0
SNR_for_im = 0
hyp_par =  [300]
 
def sigTOnoise(signal, noise):
    signal_to_noise_ratio = signal / noise
    return signal_to_noise_ratio

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img
for hyp in hyp_par:
    SNR_for_im = 0
    count = 0
    print(hyp)
    for file in os.listdir(clean_path):
        filename = os.fsdecode(file)
        image = Image.open(clean_path +'/'+filename)
        if image.mode == 'RGB':
            original = cv2.imread(clean_path +'/'+filename)
            resized = cv2.resize(original, (256,256))
            cropped = center_crop(resized , (224,224))
            noise =  np.random.normal(loc=0, scale=1, size=cropped.shape)
            total_noise = hyp * noise
            try:
                noisy = np.clip((cropped + total_noise),0,255)
                cv2.imwrite(noisy_path + '/' + filename, noisy)
                cv2.imwrite(cropped_path + '/' + filename, cropped)
                SIGNAL_for_im = np.sum(np.array(cropped)) 
                NOISE_for_im = np.sqrt(np.square(np.sum(np.array(total_noise))))
                SNR_for_im += sigTOnoise(SIGNAL_for_im, NOISE_for_im)
                count+=1
            except:
                ValueError

    print('calculating signal to noise ratio')
    signal_to_noise = 10*np.log10(SNR_for_im/count)
    print(f'SNR = {signal_to_noise}')
    for classifier in classifiers: 
        if classifier == alexnet:
            AlexNet = evaluate(cropped_path, noisy_path, classifier)
            print('alexnet')
            print(AlexNet)
        elif classifier == googlenet:
            GoogLeNet = evaluate(cropped_path, noisy_path, classifier)
            print('googlenet')
            print(GoogLeNet)
        elif classifier == resnet:
            ResNet = evaluate(cropped_path, noisy_path, classifier)
            print('resnet')
            print(ResNet)
        else:
            VggNet = evaluate(cropped_path, noisy_path, classifier)
            print('vgg')    
            print(VggNet)


