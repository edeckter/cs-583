# -*- coding: utf-8 -*-
"""
This script creates a custom generator that loads batches of images and masks.
References:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
https://medium.com/the-artificial-impostor/custom-image-augmentation-with-keras-70595b01aeac
https://albumentations.readthedocs.io/en/latest/index.html
"""
#Import packages to be used in script
import numpy as np
import cv2
from scipy.sparse import load_npz
from albumentations import (Compose, HorizontalFlip, ShiftScaleRotate)
from tensorflow.python.keras.utils.data_utils import Sequence

###########################################################
class ImageAndMaskGenerator(Sequence):
    #Generates data for Keras
    def __init__(self, image_names, augment=False, num_classes=5, batch_size=32, resize=(350,525), shuffle=True):
        self.image_names = image_names
        self.batch_size = batch_size
        self.resize = resize
        self.augment = augment
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        #Denotes the number of batches per epoch
        return int(np.floor(len(self.image_names) / float(self.batch_size)))

    def __getitem__(self, index):
        #Generate one batch of data
        #Get indices of the batch
        indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_image_names = [self.image_names[i] for i in indices]
        # Generate data
        images, masks = self.__data_generation(batch_image_names)
        
        if self.augment:
            #Define augmentations to perform on data
            augment = Compose([HorizontalFlip(p=0.5),
                               ShiftScaleRotate(shift_limit=0.2, 
                                                scale_limit=0.1,
                                                rotate_limit=40, 
                                                border_mode=cv2.BORDER_REFLECT_101, p=0.8)
                             ])
            #Perform augmentation on image and mask
            aug_images = np.empty(images.shape)
            aug_masks = np.empty(masks.shape)
            
            for i, image in enumerate(images):
                pair = {'image': images[i], 'mask': masks[i]}
                augmentation = augment(**pair)
                aug_images[i] = augmentation['image']
                aug_masks[i] = augmentation['mask']
            images = aug_images
            masks = aug_masks
        
        return images, masks

    def on_epoch_end(self):
        #Updates indexes after each epoch
        self.indexes = np.arange(len(self.image_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_image_names):
        #Generates data containing batch_size samples
        # Initialization
        images = np.zeros((self.batch_size, *self.resize, 3))
        masks = np.zeros((self.batch_size, *self.resize, self.num_classes))
        
        for i, image_name in enumerate(batch_image_names):
            path = '' #Local implementation
            #path = 'drive/My Drive/CS 583/Project/'
            try:
                img = cv2.imread(path+'data/train_images/'+image_name+'.jpg')
                mask = load_npz(path+'data/train_masks/'+image_name+'.npz').toarray()
            except:
                continue #skip to next iteration if files don't exist
            #Continue load if image and masks exist
            #Resize images and masks
            img = cv2.resize(img, dsize=(self.resize[1],self.resize[0]))
            mask = cv2.resize(mask, dsize=(self.resize[1],self.resize[0]))
            #One-hot encode mask labels
            flatten = mask.reshape(mask.shape[0]*mask.shape[1])
            one_hot = np.zeros((flatten.shape[0],self.num_classes))
            for row, value in enumerate(flatten):
                value = int(value)
                one_hot[row,value] = 1
            mask = one_hot.reshape((*mask.shape, self.num_classes))
                
            #Save image in batch
            images[i] = img
            masks[i] = mask
            
        return images, masks
    
class ImageGenerator(Sequence):
    #Generates data for Keras
    def __init__(self, image_names, batch_size=32, resize=(350,525), shuffle=False):
        self.image_names = image_names
        self.batch_size = batch_size
        self.resize = resize
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        #Denotes the number of batches per epoch
        return int(np.floor(len(self.image_names) / float(self.batch_size)))

    def __getitem__(self, index):
        #Generate one batch of data
        #Get indices of the batch
        indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_image_names = [self.image_names[i] for i in indices]
        # Generate data
        images = self.__data_generation(batch_image_names)
        
        return batch_image_names, images

    def on_epoch_end(self):
        #Updates indexes after each epoch
        self.indexes = np.arange(len(self.image_names))
        if self.shuffle == False:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_image_names):
        #Generates data containing batch_size samples
        # Initialization
        images = np.zeros((self.batch_size, *self.resize, 3))
        
        for i, image_name in enumerate(batch_image_names):
            path = '' #Local implementation
            #path = 'drive/My Drive/CS 583/Project/'
            try:
                img = cv2.imread(path+'data/test_images/'+image_name+'.jpg')
                #Resize images and masks
                img = cv2.resize(img, dsize=(self.resize[1],self.resize[0]))
                #Save image in batch
                images[i] = img
            except:
                continue #skip if file doesn't exist
            
        return images

#########Test generator#########
if __name__=='__main__':
    import pathlib
    import matplotlib.pyplot as plt
    
    #Get training image names
    image_names = [f.name[:-4] for f in (pathlib.Path.cwd() / 'data' / 'train_images').glob('*')]
    #Create training image generator with batch size of 1 and random augmentation
    generator = ImageAndMaskGenerator(image_names, resize=(1400, 2100), augment=False, batch_size=1)
    
    #Get training image and mask
    for b, batch in enumerate(generator):
        if b==1:
            break
    batch_image, batch_mask = batch
    batch_image = batch_image.reshape((batch_image.shape[1], batch_image.shape[2], batch_image.shape[3]))
    batch_mask = np.argmax(batch_mask, axis=-1)
    batch_mask = batch_mask.reshape((batch_mask.shape[1], batch_mask.shape[2]))
    mask = batch_mask
    #Show satellite cloud image
    print('Random Training Image')
    fig = plt.imshow(batch_image.astype(int))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
    #Show corresponding mask
    print('Corresponding Image Mask')
    fig = plt.imshow(batch_mask.astype(int))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
    
    #Superimpose mask onto satellite image
    batch_mask = batch_mask/4
    batch_mask = np.uint8(batch_mask * 255)
    batch_mask = cv2.applyColorMap(batch_mask, cv2.COLORMAP_JET)
    
    superimposed = cv2.addWeighted(np.uint8(batch_image), 0.6, batch_mask, 0.4, 0)
    print('Image Mask Superimposed over Training Image')
    fig = plt.imshow(superimposed)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
    
    #Get test image names
    test_images = [f.name[:-4] for f in (pathlib.Path.cwd() / 'data' / 'test_images').glob('*')]
    #Create test image generator with batch size of 1
    test_generator = ImageGenerator(test_images, batch_size=1)
    
    #Get test image
    for b, test_image in enumerate(test_generator):
        if b==1:
            break
    test_image = test_image.reshape((test_image.shape[1], test_image.shape[2], test_image.shape[3]))
    #Show satellite cloud image
    print('Random Test Image')
    fig = plt.imshow(test_image.astype(int))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
