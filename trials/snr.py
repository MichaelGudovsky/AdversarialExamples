import numpy as np
import cv2
import torch
from PIL import Image
import os

evice = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
orig_test_path = './for_graph'
numOFfiles = 0
snr = 0

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



def signaltonoise(a, axis=None, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

for file in os.listdir(orig_test_path):
    filename = os.fsdecode(file)
    image = Image.open(orig_test_path +'/'+filename)
    if image.mode == 'RGB':
        numOFfiles+=1
        original = cv2.imread(orig_test_path +'/'+filename)
        resized = cv2.resize(original, (256,256))
        cropped = center_crop(resized , (224,224)) 
        res = signaltonoise(image)
        snr += res
total_SNR = snr/numOFfiles
print(total_SNR)
