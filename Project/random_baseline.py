# -*- coding: utf-8 -*-
"""
Generate random baseline for image segmentation
and calculate Dice coefficient by average coefficient
for multiple random masks against random images
"""
#Import packages to be used in script
from generator import ImageAndMaskGenerator #Custom image generator class
import pathlib
import numpy as np

###########################################################
#Function to perform one-hot encoding by class for a 2D array
def one_hot(array, num_classes=5):
    one_hot_array = np.zeros((*array.shape,num_classes))
    for i, row in enumerate(array):
        for j, value in enumerate(row):
            one_hot_array[i,j,value] = 1
    return one_hot_array

#Generate random prediction mask for number of classses
def random_predictions(num_classes=5, image_size=(1400,2100)):
    #Generate random prediction mask 
    prediction = np.random.randint(low=0, high=num_classes, size=image_size[0]*image_size[1]).reshape((image_size[0],image_size[1]))
    #One-hot encode prediction mask
    prediction_one_hot = one_hot(prediction)
    return prediction_one_hot

#Function to calculate Dice coefficient for a single class
def dice_coefficient(y_true, y_pred):
    #By definition, return 1 for an empty category
    if (np.sum(y_true)==0) and (np.sum(y_pred)==0):
        dice = 1
    else:
        dice = (2 * np.sum((y_true * y_pred).astype(int))) / (np.sum(y_true) + np.sum(y_pred))
    return dice

#Function to calculate the average dice coefficient of across all classes
def mean_dice_coefficient(y_true, y_pred):
    #Ignore the first channel (background category)
    true = y_true[...,1:]
    pred = y_pred[...,1:]
    #Compute dice coefficient for each cloud type separately
    batch_size = true.shape[0] #Images in batch
    num_categories = true.shape[-1] #Number of cloud types (should be 4)
    mean_dice = 0
    #Loop through each example in batch
    for i in range(batch_size):
        #Loop through each category
        for j in range(num_categories):
            image_dice = dice_coefficient(true[i,...,j],pred[i,...,j])
            mean_dice += image_dice
    #Compute final mean dice score by averaging over all examples and all categories
    mean_dice /= (batch_size*num_categories)
    return mean_dice

######################################################################
if __name__=='__main__':
    #Get names of training images to be used to pull random images
    image_names = [f.name[:-4] for f in (pathlib.Path.cwd() / 'data' / 'train_images').glob('*')]
    np.random.shuffle(image_names)
    training_images = image_names

    #Set image size
    resize_images = (1400,2100)

    #Create data generator
    generator = ImageAndMaskGenerator(training_images, batch_size=1, resize=resize_images, augment=False)

    #This code calculates the Dice coefficient for a random image against a random mask
    n_iterations = 100
    dice = np.empty(shape=(n_iterations))
    #Loop for n iterations
    for b, batch in enumerate(generator):
    #Run for n iterations
        if b == n_iterations:
            break
        print('Running iteration {:d}...'.format(b))
        #Pull random image from training dataset
        _, mask = batch
        #Make random prediction for image mask
        pred_mask = random_predictions()
        #Calculate dice coefficient for this iterations
        dice_coeff = dice_coefficient(mask, pred_mask)
        dice[b] = dice_coeff
    
    #Calculate average dice coefficient
    print('Average dice coefficient: {:.2f}'.format(dice.mean()))