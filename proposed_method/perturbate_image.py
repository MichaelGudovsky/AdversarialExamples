from pydoc import describe
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
torch.manual_seed(42)
#from evaluate_model import *
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
training_data_path = '/media/michael/500GB/train' 
directory =  './test'
perturbated_images_path = './for_graph' 



mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

   
if not os.path.exists('./models_part/Proposed.npy'):  ####n to run the proposed method
#if not os.path.exists('./models_part/DF.npy'):  ####n to run the average method
# both models exist in the path
    from perturbation import perturbation
    pert = perturbation(training_data_path)
else:
    print('Found prepared perturbation')

    with open('./models_part/Proposed.npy', 'rb') as f:   #########################
        pert = np.load(f)

# iterate all files from a directory
for file in os.listdir(directory):
    try:
        old_name = os.path.join(directory, file)
        _name = old_name.split("_")[2]
        new_name = os.path.join(directory, _name)
            # Renaming the file
        os.rename(old_name, new_name)
    except:
        IndexError

  
def insert_noise(image):
    im1 = (transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = mean,std = std)])(image)).to(device)

    pert_image = 0.7 * im1 + 0.35 * torch.from_numpy(pert).cuda()
    total_noise = 0.35 * torch.from_numpy(pert).cpu()
    
    def clip_tensor(A, minv, maxv):
        tens1 = (minv*torch.ones(A.shape)).cuda()
        tens2 = (maxv*torch.ones(A.shape)).cuda()
        A = torch.max(A, tens1)
        A = torch.min(A, tens2)
        return A

    clip = lambda x: clip_tensor(x, 0, 255)
    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                            transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                            transforms.Lambda(clip),
                            transforms.ToPILImage(),
                            transforms.CenterCrop(224)])
    img = tf(pert_image[0])

    return img



# iterate over files in
# that directory
for filename in os.listdir(directory):
	# checking if it is a file
    image = Image.open(directory + '/' + filename)  # check path
    if image.mode == 'RGB':
        img = insert_noise(image)
        temp = img.save(os.path.join(perturbated_images_path , filename))
