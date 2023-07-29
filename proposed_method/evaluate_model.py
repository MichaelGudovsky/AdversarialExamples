import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
orig_test_path = './test_part'
pert_test_path =  './ready_images'  # change to ResNet
numOFfiles = 0
with open("synset_words.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device)
resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(device)
googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT).to(device)
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(device)
mobilenet = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.DEFAULT).to(device)
classifiers =  alexnet

def evaluate(orig, pert, model):

    cnt = 0
    numOFfiles = 0
    with open("synset_words.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
 
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for file in os.listdir(orig): 
        filename = os.fsdecode(file)
        im_orig = Image.open(orig +'/'+filename)
        if im_orig.mode == 'RGB':
            numOFfiles+=1
            input_tensor_o = preprocess(im_orig)
            input_batch_o = input_tensor_o.unsqueeze(0)
            im_pert = Image.open(pert +'/'+filename)
            input_tensor_p = preprocess(im_pert) 
            input_batch_p = input_tensor_p.unsqueeze(0)

            if torch.cuda.is_available():
                input_batch_o = input_batch_o.to('cuda')
                input_batch_p = input_batch_p.to('cuda')
                model.to('cuda')

            with torch.no_grad():
                output_o = model(input_batch_o)
                output_p = model(input_batch_p)

            probabilities_o = torch.nn.functional.softmax(output_o[0], dim=0)
            probabilities_p = torch.nn.functional.softmax(output_p[0], dim=0)
            _, pred_orig = torch.topk(probabilities_o, 1)
            _, pred_pert = torch.topk(probabilities_p, 1)
            if pred_orig ==pred_pert:
                cnt+=1
    fooling_success = ((cnt/numOFfiles)*100)  # (1-(cnt/numOFfiles))
    return fooling_success

evaluate(orig_test_path, pert_test_path, classifier )


