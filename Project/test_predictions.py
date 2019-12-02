# -*- coding: utf-8 -*-
"""
Script that creates the submission.csv file for the Kaggle competition
Predicts masks for test images and then converts to encoded pixel format
"""
#If using colab, run the following commands
#Mount Google Drive
'''
from google.colab import drive
drive.mount('/content/drive')

#Run only for FCN model!!
#Update tensorflow to v2
!pip install --upgrade tensorflow
#Update keras to latest
!pip install --upgrade keras
'''
#Import packages to be used in script
from generator import ImageGenerator
from mask_rcnn_config import InferenceConfig
from random_baseline import random_predictions
from tensorflow.keras.models import load_model
import pathlib
import cv2
import numpy as np
import math
from PIL import Image
from mrcnn.model import MaskRCNN

#############################################################
#Make model predictions on test dataset
def prediction_masks(images, model_option):
    if model_option=='random':
        predictions = np.zeros((*images.shape[0:3],5))
        for i, image in enumerate(images):
            predictions[i] = random_predictions(image_size=(350,525))
    elif model_option=='fcn':
        #Local path
        model_path = 'image_segmentation.h5'
        #Colab path
        #model_path = 'drive/My Drive/CS 583/Project/image_segmentation.h5'
        model = load_model(model_path)
        predictions = model.predict(images)
    elif model_option=='mask-rcnn':
        #Recreate the model in inference mode
        inference_config = InferenceConfig()
        model = MaskRCNN(mode='inference', config=inference_config, model_dir='./')
        #Local path
        model_path = 'mask_rcnn_clouds_config_0001.h5'
        #Colab path
        #model_path = 'drive/My Drive/CS 583/Project/mask_rcnn_clouds_config_0001.h5'
        #Load trained weights
        model.load_weights(model_path, by_name=True)
        samples = np.zeros((images.shape[0],1024,1024,images.shape[-1]))
        for i, name in enumerate(names):
            image = Image.open('data/test_images/'+name)
            image = image.resize((1024,682))
            new_im = Image.new("RGB", (1024,1024))
            samples[i] = new_im.paste(image, ((0,170)))
        predictions = model.detect(samples, verbose=0)
        for i, pred in enumerate(predictions):
          pred_mask = predictions[i]['masks']
          predictions[i] = pred_mask[i][170:854,:] #Trim off top and bottom padding
    else:
        print('Error: No such model option.')
    return predictions

#Write pixel encoded results to csv file
def create_submission(test_images, file, model_option):
    names, images = test_images
    #Predict image segmentation for test batch
    pred_masks = prediction_masks(images, model_option)
    #Mappings from integer indices to cloud labels
    reverse_dictionary = {0:'None', 1:'Fish', 2:'Flower', 3:'Gravel', 4:'Sugar'}
    #Get prediction masks for each image in batch
    for i, name in enumerate(names):
        print('Encoding Image {:d} of {:d}...'.format(i+1, len(names)))
        test_mask = pred_masks[i].reshape(*pred_masks.shape[1:])
        #Get classes for each pixel
        test_mask = np.argmax(test_mask, axis=-1).astype('float32')
        #Resize to output shape expected by contest
        output_shape = (350, 525)
        test_mask = cv2.resize(test_mask, dsize=(output_shape[1], output_shape[0]))
        #Flatten test mask predictions
        flat_mask = test_mask.reshape(test_mask.shape[0]*test_mask.shape[1], order='F')

        #Determine run lengths of each type of cloud
        for label in np.arange(1,5):
            line = str(name)+'_'+reverse_dictionary.get(label)+','
            position = 0
            run = 0
            for i, value in enumerate(flat_mask):            
                if (value == label) and (position == 0):
                    position = i + 1
                    run += 1
                elif (value == label) and (run != 0):
                    run += 1
                elif (value != label) and (run != 0):
                    line = line + str(position)+' '+str(run)+' '
                    position = 0
                    run = 0
            #Write pixel runs to submission file
            print(line)
            file.write(line+'\n')
################################################################
if __name__=='__main__':
    #Get list of test images
    test_images = [f.name for f in (pathlib.Path.cwd() / 'data' / 'test_images').glob('*')]
    print('Test Image Count: {:d}'.format(len(test_images)))

    model_option = 'fcn'
    #Resize images to be fed into model
    if model_option=='random':
        resize_images = (350,525) #Random baseline resize
    elif model_option=='fcn':
        resize_images = (448,672) #FCN resize
    elif model_option=='mask-rcnn':
        resize_images = (1400,2100) #Resize is handled by config
    else:
        print('Please choose a valid model option')
    
    batch_size = math.ceil(len(test_images) / 100)
    #Use for random baseline and FCN
    test_generator = ImageGenerator(test_images, batch_size=batch_size, resize=resize_images)

    #Open file for writing
    f = open('submission.csv', 'w+')
    f.write('Image_Label,EncodedPixels\n')
    
    #Write pixel encoded image masks out to file in batches
    stop_flag = False
    iteration = 1
    for b, batch in enumerate(test_generator):
        while not stop_flag:
            if iteration*batch_size > len(test_images):
                batch = batch[0:(len(test_images)-iteration*batch_size)] 
                stop_flag = True
            #Write predictions to submission file
            print('Running Batch {:d}'.format(iteration))
            create_submission(batch, f, model_option=model_option)
            iteration += 1

    #Close file after last batch
    f.close()