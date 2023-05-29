from skimage.metrics import structural_similarity
import cv2
import torch
from PIL import Image
import os
print('Calculating Structual Similarity Index')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
orig_test_path = './test_part'
pert_test_path =  './ready_images'  # change to ResNet
numOFfiles = 0

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

# def scale_image(img, factor=1):
# 	"""Returns resize image by scale factor.
# 	This helps to retain resolution ratio while resizing.
# 	Args:
# 	img: image to be scaled
# 	factor: scale factor to resize
# 	"""
# 	return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))




def similarity(orig, pert):
    similarity = 0
    numOFfiles = 0
    cnt = 0
    for file in os.listdir(orig_test_path):
        numOFfiles+=1
        filename = os.fsdecode(file)
        image = Image.open(orig +'/'+filename)
        if image.mode == 'RGB':
            original = cv2.imread(orig +'/'+filename)
            resized = cv2.resize(original, (256,256))
            cropped = center_crop(resized , (224,224)) 
            perturbated = cv2.imread(pert +'/'+filename) 
    # Convert images to grayscale
            before_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.cvtColor(perturbated, cv2.COLOR_BGR2GRAY)
            try:
                (score, diff) = structural_similarity(before_gray, after_gray, full=True)
                similarity +=score
                cnt +=1  
                # cv2.imshow('before', before)
                # cv2.imshow('after', after)
                # cv2.imshow('diff',diff)
                #cv2.waitKey(0)
            except:
                 ValueError

    similarity_over_test_set = similarity/cnt
    print(similarity_over_test_set )
    return similarity_over_test_set 
similarity(orig_test_path, pert_test_path)





# before = cv2.imread('./test/123.JPEG')
# after = cv2.imread('./images/pert.JPEG')

# # Convert images to grayscale
# before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
# after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# # Compute SSIM between two images
# (score, diff) = structural_similarity(before_gray, after_gray, full=True)
# print("Image similarity", score)

# # The diff image contains the actual image differences between the two images
# # and is represented as a floating point data type in the range [0,1] 
# # so we must convert the array to 8-bit unsigned integers in the range
# # [0,255] before we can use it with OpenCV
# diff = (diff * 255).astype("uint8")

# # Threshold the difference image, followed by finding contours to
# # obtain the regions of the two input images that differ
# thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]

# mask = np.zeros(before.shape, dtype='uint8')
# filled_after = after.copy()

# for c in contours:
#     area = cv2.contourArea(c)
#     if area > 40:
#         x,y,w,h = cv2.boundingRect(c)
#         cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
#         cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
#         cv2.drawContours(mask, [c], 0, (0,255,0), -1)
#         cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

