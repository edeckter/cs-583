# -*- coding: utf-8 -*-
"""
This script builds the image segmentation model using FCN-8 architecture for the 
Understanding Clouds from Satellite Images (https://www.kaggle.com/c/understanding_cloud_organization) Kaggle Competition.
References:
https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html
https://nanonets.com/blog/how-to-do-semantic-segmentation-using-deep-learning/
"""
'''
Run if using Google Colab
from google.colab import drive
drive.mount('/content/drive')

#Upgrade tensorflow to v2.0
!pip install --upgrade tensorflow==2
#Upgrade keras to latest
!pip install --upgrade keras
'''
#Import packages/libraries
from generator import ImageAndMaskGenerator #Custom image generator class
import pathlib
import numpy as np
import math
from keras.applications import VGG16
from keras.models import Model
from keras import layers
from keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks

################################################################
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
    #Compute dice coefficient for each cloud type separately
    batch_size = len(y_true) #Images in batch
    mean_dice = 0
    #Loop through each example in batch
    for i in range(batch_size):
        #Ignore the first channel (background category)
        true = y_true[i][...,1:]
        pred = y_pred[i][...,1:]
        num_categories = true.shape[-1] #Number of cloud types (should be 4)
        #Loop through each category
        for j in range(num_categories):
            image_dice = dice_coefficient(true[...,j],pred[...,j])
            mean_dice += image_dice
    #Compute final mean dice score by averaging over all examples and all categories
    mean_dice /= (batch_size*num_categories)
    return mean_dice

#Calculate mean dice coefficient
def evaluate_model(generator, batch_size, model):
    gt_masks = []
    predictions = []

    iteration = 1
    stop_flag = False
    for b, batch in enumerate(generator):
        while not stop_flag:
            images, masks = batch
            if iteration*batch_size > len(validation_images):
                images = images[0:(len(validation_images)-iteration*batch_size)] 
                masks = masks[0:(len(validation_images)-iteration*batch_size)] 
                stop_flag = True
            print('Iteration {:d}'.format(iteration))
            #Save ground truth to list
            gt_masks.append(masks)
            #Make predictions
            prediction = model.predict(images)
            #Extract results for first sample
            predictions.append(prediction)
            iteration += 1
            
    #Calculate mean Dice coefficient
    mean_dice = mean_dice_coefficient(gt_masks, predictions)
    return mean_dice

##################################################################################
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

    #Create train and validation data generators
    resize_images = (448,672)
    train_generator = ImageAndMaskGenerator(training_images, batch_size=22, resize=resize_images, augment=True)
    validation_generator = ImageAndMaskGenerator(validation_images, batch_size=22, resize=resize_images, augment=False)

    ##Build FCN-8 Model
    #Use VGG16 pre-trained on ImageNet as the encoder
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(*resize_images,3))
    #Freeze all layers in VGG16 pre-trained model
    for layer in conv_base.layers:
        layer.trainable=False
        
    #Outputs for skip connections
    for layer in conv_base.layers:
        if (layer.name=='block4_pool'):
            pool4_output = layer.output
        if (layer.name=='block3_pool'):
            pool3_output = layer.output
        if (layer.name=='block1_pool'):
            pool1_output = layer.output
                        
    #Create upsampling section of FCN
    decode = layers.Deconvolution2D(512, kernel_size=(1,1), activation='relu', padding='same', name='1x1_conv')(conv_base.output)
    decode = layers.Dropout(0.5)(decode)
    decode = layers.Deconvolution2D(512, kernel_size=(3,3), activation='relu', padding='same', name='decode_block1_conv')(decode)
    decode = layers.UpSampling2D((2, 2), name='decode_block1_upsample')(decode)
    decode = layers.add([decode, pool4_output]) #Add skip connection
    decode = layers.Dropout(0.5)(decode)
    decode = layers.Deconvolution2D(256, kernel_size=(3,3), activation='relu', padding='same', name='decode_block2_conv')(decode)
    decode = layers.UpSampling2D((2, 2), name='decode_block2_upsample')(decode)
    decode = layers.add([decode, pool3_output]) #Add skip connection
    decode = layers.Dropout(0.5)(decode)
    decode = layers.Deconvolution2D(64, kernel_size=(3,3), activation='relu', padding='same', name='decode_block3_conv')(decode)
    decode = layers.UpSampling2D((4, 4), name='decode_block3_upsample')(decode)
    decode = layers.add([decode, pool1_output])
    decode = layers.Dropout(0.5)(decode)
    decode = layers.Deconvolution2D(3, kernel_size=(3,3), activation='relu', padding='same', name='decode_block4_conv')(decode)
    decode = layers.UpSampling2D((2, 2), name='decode_block4_upsample')(decode)
    decode = layers.add([decode, conv_base.input])
    output = layers.Conv2D(5, kernel_size=(1,1), activation='softmax', padding='same', name='output_layer')(decode)

    #Build encoder/decoder model
    model = Model(inputs=conv_base.inputs, outputs=output, name='cnn_image_segmentation')
    plot_model(model, show_shapes=True, to_file='FCN.png')
    print(model.summary())

    #Create callback to monitor model training
    #Use early stopping if validation loss doesn't improve for 3 consecutive epochs and save best model during training

    #Google Drive Path
    #path = 'drive/My Drive/CS 583/Project/image_segmentation.h5'
    #Local Path
    path = 'image_segmentation.h5'
    
    #Create model callbacks
    model_callbacks = [callbacks.EarlyStopping(monitor='acc', mode='max', patience=1),
                       callbacks.ModelCheckpoint(filepath=path,
                                                 monitor='val_acc',
                                                 mode='max',
                                                 save_best_only=True)
                       ]

    #Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    
    #Fit model
    model.fit_generator(train_generator,
                        steps_per_epoch=200,
                        epochs=10,
                        callbacks=model_callbacks,
                        validation_data=validation_generator,
                        validation_steps=50)

    #Load pre-trained model weights
    model_path = 'image_segmentation.h5'
    #Colab path
    #model_path = 'drive/My Drive/CS 583/Project/image_segmentation.h5'
    model = load_model(model_path)
    #Evaluate model
    batch_size = 22
    print('Training Set Dice Coefficient {:.2f}'.format(evaluate_model(train_generator, batch_size, model)))
    print('Validation Set Dice Coefficient {:.2f}'.format(evaluate_model(validation_generator, batch_size, model)))