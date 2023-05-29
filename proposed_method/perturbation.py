import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from deepfool import *
import os
orig = './models_temp'
def perturbation(path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
    directory = path
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    count = 0

    '''
    addition for saving checkpoints of the model
    '''

    def save_checkpoint_model(counter):
        cnt = str(counter)

        with open(orig + '/' + cnt + '.npy', 'wb') as s:
            np.save(s, pert)
            print(f"saved", {cnt})

        return

    #classifiers = https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device)
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(device)
    googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT).to(device)
    vgg = models.vgg11(weights=models.VGG11_Weights.DEFAULT).to(device)
    mobilenet = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.DEFAULT).to(device)

    classifiers =  [alexnet] #[alexnet, resnet, googlenet, vgg] #[alexnet, resnet, vgg]
    listOfFiles = list()

    for (dirpath, dirnames, filenames) in os.walk(directory):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    for classifier in classifiers:
        net = classifier
    # Switch to evaluation mode
        net.eval()
        #print(f"training on {net}")
        for file in listOfFiles:
        # checking if it is a file
            im_orig = Image.open(file)
            if im_orig.mode == 'RGB':
                im = (transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = mean,std = std)])(im_orig))#.to(device)
                
                if count == 0:
                    r,_, _, _, _ = (deepfool(im, net))
                    pert = r
                else:
                    new_im = im + pert
                    new_im = torch.squeeze(new_im,0)
                    new_im.to(device)
                    r,_, _, _, _ = (deepfool(new_im, net))
                    pert = pert + r
                count+=1


        print('learning done')
    
    with open('./models_part/Proposed.npy', 'wb') as f:  #################################################
        np.save(f, pert)
        print('model saved')


    return pert