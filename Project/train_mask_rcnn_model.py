# -*- coding: utf-8 -*-
"""
This script builds the image segmentation model using Mask R-CNN architecture for the 
Understanding Clouds from Satellite Images (https://www.kaggle.com/c/understanding_cloud_organization) Kaggle Competition.
References:
https://github.com/matterport/Mask_RCNN
https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/
"""
'''
Run if using Google Colab
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

#Set path
import os
path = 'drive/My Drive/CS 583/Project/'
os.chdir(path)

#Install Mask RCNN package from github
#if not os.path.exists(path+'Mask_RCNN'):
!git clone https://github.com/matterport/Mask_RCNN.git
os.chdir('Mask_RCNN/')
!python setup.py install
os.chdir('..')
'''
#Import packages
import numpy as np
import math
import pathlib
import imgaug
from mrcnn.model import MaskRCNN
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn import visualize
from mrcnn.model import log
from mask_rcnn_config import CloudsDataset, CloudsConfig, InferenceConfig

#Function to calculate Dice coefficient
def dice_coefficient(y_true, y_pred):
    #By definition, return 1 for an empty category
    if (np.sum(y_true)==0) and (np.sum(y_pred)==0):
        dice = 1
    else:
        dice = (2 * np.sum((y_true * y_pred).astype(int))) / (np.sum(y_true) + np.sum(y_pred))
    return dice

def mean_dice_coefficient(y_true, true_class_ids, y_pred, pred_class_ids, num_classes):
    #Compute dice coefficient for each cloud type separately
    batch_size = len(y_true) #Images in batch
    mean_dice = 0
    #Loop through each example in batch
    for i in range(batch_size):
        #Loop through each category
        for j in range(num_classes):
          try:
            t = np.where(true_class_ids[i]==j)
            true = y_true[i][...,t]
          except:
            true = np.zeros((*y_true[i].shape[0:-1]))
          try:
            p = np.where(pred_class_ids[i]==j)
            pred = y_pred[i][...,p]
          except:
            pred = np.zeros((*y_pred[i].shape[0:-1]))
          
          image_dice = dice_coefficient(true, pred)
          mean_dice += image_dice
    #Compute final mean dice score by averaging over all examples and all categories
    mean_dice /= (batch_size*num_classes)
    return mean_dice

#Calculate mean dice coefficient
def evaluate_model(dataset, model, config):
    gt_masks = []
    gt_class_ids = []
    predictions = []
    pred_class_ids = []
    for i, image_id in enumerate(dataset.image_ids):
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset_val, config, image_id, use_mini_mask=False)
        #Save ground truth to list
        gt_masks.append(gt_mask)
        gt_class_ids.append(gt_class_id)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, config)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        prediction = model.detect(sample, verbose=0)
        # extract results for first sample
        predictions.append(prediction[0]['masks'])
        pred_class_ids.append(prediction[0]['class_ids'])
	
    #Calculate mean Dice coefficient
    mean_dice = mean_dice_coefficient(gt_masks, gt_class_ids, predictions, pred_class_ids, num_classes=4)
    return mean_dice

############################################################################
if __name__=='__main__':
    #Split training dataset into training and validation datasets
    image_names = [f.name[:-4] for f in (pathlib.Path.cwd() / 'data' / 'train_images').glob('*')]
    images = len(image_names)
    np.random.seed = 42
    training_images = np.random.choice(image_names, math.floor(images*0.9), replace=False)
    validation_images = [image for image in image_names if image not in training_images]

    #Show training and validation dataset size
    print('Training Image Count: {:d}'.format(len(training_images)))
    print('Validation Image Count: {:d}'.format(len(validation_images)))

    #Define augmentation criteria
    image_augmentation = imgaug.augmenters.SomeOf((0, None), [
                          imgaug.augmenters.Affine(rotate=(-40,40)),
                          imgaug.augmenters.Affine(scale=(0.5, 1.5)),
                          imgaug.augmenters.Fliplr(0.5)
                          ])

    #Load data
    # Training dataset
    dataset_train = CloudsDataset()
    dataset_train.load_batch(training_images)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CloudsDataset()
    dataset_val.load_batch(validation_images)
    dataset_val.prepare()

    #Prepare configuration
    config = CloudsConfig()

    #Create model in training model
    model = MaskRCNN(mode='training', model_dir='./', config=config)
    #Load pre-trained weights (MSCOCO) and exclude the output layers
    model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    #Train weights (output layers or 'heads')
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads', augmentation=image_augmentation)
    
    #Create instance of inference configuration class
    inference_config = InferenceConfig()

    #Recreate the model in inference mode
    model = MaskRCNN(mode='inference', config=inference_config, model_dir='./')

    #Get path to saved weights
    model_path = 'mask_rcnn_clouds_config_0001.h5'

    #Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    #Test on random validation image
    image_id = np.random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset_train.class_names, figsize=(8, 8))

    #convert image into one sample
    sample = np.expand_dims(mold_image(original_image, inference_config), 0)
    model_prediction = model.detect(sample, verbose=1)

    #Evaluate model performance on training and validation dataset
    print('Training Set Dice Coefficient {:.2f}'.format(evaluate_model(dataset_train, model, inference_config)))
    print('Validation Set Dice Coefficient {:.2f}'.format(evaluate_model(dataset_val, model, inference_config)))