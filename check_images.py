import os
from os import listdir
from PIL import Image

# This script can detect crropted images in the dataset
# If there's any corrpted image, PIL will raise OSError during training


detaset_path = 'action_effect_images'

for foldername in listdir(detaset_path):
  sign_path = os.path.join(detaset_path, foldername)
#   print(sign_path)
  for sign in listdir(sign_path):
    image_path = os.path.join(sign_path, sign)
    try:
        img = Image.open(image_path) # open the image file
        img.verify() # verify that it is, in fact an image
    except (IOError, SyntaxError) as e:
        print('Bad file:', image_path) # print out the names of corrupt files