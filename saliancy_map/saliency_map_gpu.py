######################################################################################
#########################################################################################
import torch
import cv2
import torchvision
import  torchvision.transforms as transforms
import numpy as np
import os 
from PIL import Image
torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
directory = '/home/michael/VScode/deep_fool/test'
perturbated_images_path = './home/michael/VScode/deep_fool/saliency/'
def saliency_map(image):
    #load pretrained resnet models
    alexnet = torchvision.models.alexnet(weights = torchvision.models.AlexNet_Weights.DEFAULT).to(device)
    resnet = torchvision.models.resnet101(weights = torchvision.models.ResNet101_Weights.DEFAULT).to(device)
    vgg = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT).to(device)
    googlenet = torchvision.models.googlenet(weights = torchvision.models.GoogLeNet_Weights.DEFAULT).to(device)
    classifiers = [alexnet]
    #define transforms to preprocess input image into format expected by model
    #resize image to the size expected by pretrained model,
    #convert PIL image to tensor, and normalize the image
    transform = (transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)        
    ]))

    def saliency(img, model):
        #we don't need gradients w.r.t. weights for a trained model
        for param in model.parameters():
            param.requires_grad = False
        #set model in evaluation mode
        model.eval()
        model = model
        #transoform input PIL image to torch.Tensor and normalize
        input = transform(img)
        input = input.to(device)
        input.unsqueeze_(0)
        #we want to calculate gradient of higest score w.r.t. input
        #so set requires_grad to True for input 
        input.requires_grad = True
        #forward pass to calculate predictions
        preds = model(input).cuda()
        score, indices = torch.max(preds, 1)
        #backward pass to get gradients of score predicted class w.r.t. input image
        score.backward()
        #get max along channel axis
        slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
        #normalize to [0..1]
        slc = (slc - slc.min())/(slc.max()-slc.min())
        slc = slc.reshape(1, 224, 224).repeat(3, 1, 1)
        return slc


    def clip_tensor(A, minv, maxv):
        
        tens1 = (minv*torch.ones(A.shape)).to(device)  #.cuda() #.cpu()
        tens2 = (maxv*torch.ones(A.shape)).to(device)  #.cuda() #.cpu()
        A = torch.max(A, tens1)
        A = torch.min(A, tens2)
        return A
    clip = lambda x: clip_tensor(x, 0, 255)
    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                            transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                            transforms.Lambda(clip),
                            transforms.ToPILImage(),
                            transforms.CenterCrop(224)])
    count = 0
    #rand_tensor = 2*torch.rand(3, 224, 224 ).cpu() - torch.ones(3, 224, 224).cpu()
    for classifier in classifiers:
        model = classifier
        sal = saliency(image_orig, model)
        if count == 0:
            pert = sal
        else:
            pert = pert + sal
        count+=1
    pert_image =  image*(1 - 2.5*pert).cuda() #cpu()
    img = tf(pert_image)
    return img
listOfFiles = list()


total_files = 0
for filename in os.listdir(directory):
	# checking if it is a file
    print(filename)
    image_orig = Image.open(directory + '/' + filename)  # check path
    if image_orig.mode == 'RGB':
        image_tensor = (transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.GaussianBlur(kernel_size = 7, sigma=(0.1, 1)),
                                    transforms.Normalize(mean = mean,std = std)])(image_orig)).to(device)
        noisy = saliency_map(image_tensor)
        noisy.save(os.path.join(perturbated_images_path , filename))

    total_files+=1
    if total_files % 1000 == 0:
        print(total_files)

