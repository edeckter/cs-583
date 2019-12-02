# -*- coding: utf-8 -*-
'''
This script converts the "train.csv" file from encoded pixels to masks
that are the same size as the training images.
The masks are saved as numpy arrays to disk to be used in the modeling.
'''

#Import packages/libraries
import pathlib
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import matplotlib.pyplot as plt
import cv2

#Functions that will be called later in the script
#Get pixels for each data type
def get_mask(train_labels, label_dict):
    #Creates blank mask of all zeros
    mask = np.zeros(1400*2100)
    #Checks if pixels exist for a given category/class of cloud
    labels = train_labels[~pd.isnull(train_labels['EncodedPixels'])]
    print(labels)
    print(labels.index)
    for i in labels.index:
        label = labels.loc[i]
        label_type = label['Label']
        pixel_list = label['EncodedPixels'].split(' ')
        pixel_list = [int(val) for val in pixel_list]
        #Set pixel values in mask to category type
        for p in np.arange(len(pixel_list), step=2):
            start_pixel = pixel_list[p]
            end_pixel = pixel_list[p]+pixel_list[p+1]
            mask[start_pixel:end_pixel] = label_dict.get(label_type)
    #Reshape mask to image shape/size (1400 x 2100)
    mask = mask.reshape((1400, 2100), order='F')
    return csr_matrix(mask)

################################################################
if __name__=='__main__':
    #Load in training label csv file
    train_labels = pd.read_csv(pathlib.Path.cwd() / 'data' / 'train.csv')
    #Split image name and cloud category/type into new columns
    splits = [text.split('_') for text in train_labels['Image_Label']]
    train_labels['Image'] = [val[0] for val in splits]
    train_labels['Label'] = [val[1] for val in splits]

    #Map label names to integer indices
    label_dict = {'None': 0, 'Fish': 1, 'Flower': 2, 'Gravel': 3, 'Sugar': 4}

    #Count unique images in training dataset
    image_count = len(train_labels['Image'].unique())
    print('There are {:,} training images in the dataset'.format(image_count))

    #Create list of unique image file names
    image_names = train_labels['Image'].unique()
       
    #Create masks for each image in the dataset
    for name in image_names:
        mask_name = str(pathlib.Path.cwd() / 'data' / 'train_masks')+'/'+name[:-4]+'.npz'
        image_mask = get_mask(train_labels[train_labels['Image']==name], label_dict)
        save_npz(mask_name, image_mask)
    
    #Check saved image masks
    image = np.random.choice(image_names, 1)[0][:-4]
    image_path = str(pathlib.Path.cwd() / 'data' / 'train_images' / image)+'.jpg'
    mask_path = str(pathlib.Path.cwd() / 'data' / 'train_masks' / image)+'.npz'
    test_image = cv2.imread(image_path)
    test_mask = load_npz(mask_path).toarray()
    plt.imshow(test_image)
    plt.show()
    plt.imshow(test_mask)
    plt.show()
    #Superimpose test mask on test image
    test_mask /= test_mask/4
    test_mask = np.uint8(test_mask * 255)
    test_mask = cv2.applyColorMap(test_mask, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(test_image, 0.6, test_mask, 0.4, 0)
    plt.imshow(superimposed)
    plt.show()