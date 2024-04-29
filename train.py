##edit this root path according to how you have placed it on your drive
Root_Dir='/home/musmani/Road Task-23'

import os
Model_Dir=os.path.join(Root_Dir, "Model" )
Weights_Dir=os.path.join(Root_Dir, "Weights" )
Dataset_Dir=os.path.join(Root_Dir, "DatasetPatched" )
os.chdir(Root_Dir)
from DatasetManager import getTrainValDatasetGenarators, getTestDatasetGenarator
from VisualizationManager import *
from models import createModel
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import pickle
image_dimensions=(512, 512, 6)
with tf.device('/device:GPU:0'):
    #create dataset generators
    trainGen,ValGen=getTrainValDatasetGenarators(Dataset_Dir, image_dimensions = image_dimensions, batch_size=16, IsVisualization=False, ApplyMaskEnhancment=True, applyAug=False)

    for architechtureName in [ "UNet" ,"AttentionResUNet", "Attention_UNet" ]:
        for lossName in ["binary_crossentropy" ,"binary_focal" , "binary_dice"]:
            model_name = architechtureName+"__"+lossName
            #get model
            model = createModel(architechtureName,lossName, image_dimensions, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True)
            #create architechture plot
            plot_model(model,show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True,to_file=os.path.join(Weights_Dir,model_name+"_plot.png"))
            #create early stopping checkpoints
            early = EarlyStopping(monitor = 'val_loss', mode = 'max', patience=10, restore_best_weights=True)
            #create learning rate reduction checkpoints
            learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 10, verbose=1,factor=0.1, min_lr=0.000001)
            #create best weights saver checkpoints
            checkpointer = ModelCheckpoint(filepath =os.path.join(Weights_Dir, model_name+"_weights.h5" ), monitor='val_IOU@0.5', verbose=1, save_best_only=True, mode='max')
            #create callback list
            callbacks_list = [ learning_rate_reduction,checkpointer]
            #train model
            history = model.fit(
                        trainGen,
                        steps_per_epoch=len(trainGen),
                        epochs=150,
                        validation_data=ValGen,
                        validation_steps=len(ValGen),
                        callbacks=callbacks_list,
                        )
            #save history
            with open(os.path.join(Weights_Dir,model_name+"_history.pkl"), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)


