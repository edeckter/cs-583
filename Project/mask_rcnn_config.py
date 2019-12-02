# -*- coding: utf-8 -*-
"""
Configuration file for Mask R-CNN model.
This file creates a Dataset class for the cloud data.
It also creates configuration classes for the training and inference models.
"""
#Import packages
import cv2
import numpy as np
from scipy.sparse import load_npz
import math
import pathlib
from mrcnn import utils
from mrcnn.config import Config
from mrcnn import visualize

#Class that defines and loads the clouds dataset
class CloudsDataset(utils.Dataset):
    #Load batch
    def load_batch(self, image_names):
      #Define cloud formation classes
      #Map label names to integer indices
      clouds = ['Fish', 'Flower', 'Gravel', 'Sugar']
      for i, cloud in enumerate(clouds):
        self.add_class('clouds', i, cloud)

		  #Get image
      for i, image_name in enumerate(image_names):
        image_id = image_name
        self.add_image('clouds', image_id=image_name, classes=clouds,
                        path='data/train_images/'+image_name+'.jpg',
                        mask='data/train_masks/'+image_name+'.npz')
    
    #Get training dataset info   
    def load_image(self, image_id):
      info = self.image_info[image_id]
      #Define box file location
      image_path = info['path']
      img = cv2.imread(image_path)

      return img
 
	#load the masks for an image
    def load_mask(self, image_id):
      info = self.image_info[image_id]
      #Define mask file location
      mask_path = info['mask']
      mask = load_npz(mask_path).toarray()

      #One-hot encode mask labels
      num_classes = len(info['classes'])
      one_hot = np.zeros((*mask.shape, num_classes))
      for row in range(mask.shape[0]):
        for column, value in enumerate(mask[row,:]):
          value = int(value)
          if value == 0:
            continue
          else:
            one_hot[row,column,value-1] = 1
			
      #Map class names to class IDs
      class_ids = np.array([self.class_names.index(c) for c in info['classes']], dtype='int32')
      return one_hot, class_ids
 
	# load an image reference
    def image_reference(self, image_id):
      info = self.image_info[image_id]
      return info
	
#Define a configuration for the model
class CloudsConfig(Config):
  #Give the configuration a recognizable name
  NAME = "clouds_config"
  #Number of classes (background + cloud formation classes)
  NUM_CLASSES = 1 + 4
  #Number of training steps per epoch
  STEPS_PER_EPOCH = 2496
  #Number of validation steps at each epoch
  VALIDATION_STEPS = 100
  #Set mini-mask configuratino
  USE_MINI_MASK = True
  MINI_MASK_SHAPE = (512, 512)  # (height, width) of the mini-mask
  #Image resizing
  IMAGE_RESIZE_MODE = 'square'
  IMAGE_MIN_DIM = 800
  IMAGE_MAX_DIM = 1024
  
#Create prediction configuration class
class InferenceConfig(CloudsConfig):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
################################################################
if __name__=='__main__':
    #Split training dataset into training and validation datasets
    image_names = [f.name[:-4] for f in (pathlib.Path.cwd() / 'data' / 'train_images').glob('*')]
    images = len(image_names)
    np.random.seed = 42
    training_images = np.random.choice(image_names, math.floor(images*0.9), replace=False)

    #Show training and validation dataset size
    print('Training Image Count: {:d}'.format(len(training_images)))
    
    #Load data
    #Training dataset
    dataset_train = CloudsDataset()
    dataset_train.load_batch(training_images)
    dataset_train.prepare()

    #Prepare configuration
    config = CloudsConfig()

    #Display test images    
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)