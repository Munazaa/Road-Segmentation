import os
import json
import cv2
import numpy as np
import sklearn.model_selection
from imgaug.augmentables import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa
import imgaug as ia
import tensorflow as tf
from keras.applications.nasnet import  preprocess_input


MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

def hook(images, augmenter, parents, default):
  """Determines which augmenters to apply to masks."""
  return augmenter.__class__.__name__ in MASK_AUGMENTERS

def GetAvailableDatasetFiles(Dir):
    """create list of available annotated dataset files"""
    filepaths=[]
    imagesNames=os.listdir(os.path.join(Dir,"images"))
    for imagename in imagesNames:
        filepaths.append([os.path.join(Dir,"images",imagename),os.path.join(Dir,"otherchannels",imagename),os.path.join(Dir,"masks",imagename)])
    
    return filepaths

def getXY(filepaths):
    images=[]
    masks=[]
    for row in filepaths:
        images.append((row[0],row[1]))
        masks.append(row[-1])
    return images, masks

# def cvtImage(image):
#     '''Applies all combinations of r, g, b an create 6 different images from 1 image'''
#     images=[]
#     for comb in [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]:
#         images.append(image[:,:,comb])
#     return images


# def IncreaseDataWithRGBComb(datasetImages,pointList):
#     '''this method takes the dataset as input and applies different rgb ordering combinations to increase the dataset'''
#     NewDatasetImages=[]
#     NewDatasetPoints=[]
#     for image, points in zip(datasetImages,pointList):
#         images=cvtImage(image)
#         for img in images:
#             NewDatasetImages.append(img)
#             NewDatasetPoints.append(points)

def ApplyDialationAndBlurThresh(m):
    kernel = np.ones((2,2),np.uint8)
    m = cv2.dilate(m,kernel,iterations = 3)
    m = cv2.GaussianBlur(m, (11,11), 0)
    # m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    ret, m = cv2.threshold(m, 0.5, 1, cv2.THRESH_BINARY)
    return m
    # return  NewDatasetImages, NewDatasetPoints
def getAugmentor():
    #create dataset augmenter
    return iaa.Sometimes(5/6,iaa.SomeOf((1, 3),[
                    #iaa.AdditiveGaussianNoise(scale=0.05 * 255, name="AWGN"),
                    # iaa.GaussianBlur(sigma=(0.0, 2.0), name="Blur"),
                    # iaa.Dropout([0.0, 0.05], name='Dropout'), # drop 0-5% of all pixels
                    #iaa.Fliplr(1),
                    # iaa.Flipud(1),
                    iaa.ChannelShuffle(1),
                    iaa.Add((-10, 10),name="Add"),
                    
                    iaa.Multiply((0.8, 1.2), name="Multiply"),
                    # iaa.Affine(scale=(1, 1.2),cval=(0, 0)),
                    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},cval=(0, 0)),
                    iaa.Affine(rotate=(-20, 20),cval=(0, 0)),  # rotate by -45 to 45 degrees
                    iaa.Affine(shear=(-20, 20),cval=(0, 0)),
                    iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
                    # iaa.PerspectiveTransform(scale=(0.01, 0.1)),
                    iaa.LinearContrast((0.8,1.2)),
                    # improve or worsen the contrast
                #     iaa.Grayscale(alpha=(0.0, 0.5)),
                 ], random_order=True))



class DataGenerator(tf.keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, images_paths, labels, batch_size=64, image_dimensions = (256, 256, 6), shuffle=True, augmentation=None,IsVisualization=False, ApplyMaskEnhancment=False):
		self.labels              = labels              # array of labels
		self.images_paths        = images_paths        # array of image paths
		self.dim                 = image_dimensions    # image dimensions
		self.batch_size          = batch_size          # batch size
		self.shuffle             = shuffle             # shuffle bool
		self.ApplyMaskEnhancment = ApplyMaskEnhancment
		self.IsVisualization     = IsVisualization
		
		if augmentation==None:
			self.augment      = False             # augment data bool
		else:
			self.augment      = True 
		self.seq=augmentation
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.images_paths) / self.batch_size))

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.images_paths))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __getitem__(self, index):
		'Generate one batch of data'
		# selects indices of data for next batch
		indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

		# select data and load images
		labels = [cv2.imread(self.labels[k]) for k in indexes]
        
        #read images
		images = [cv2.imread(self.images_paths[k][0]) for k in indexes]

        #read channels
		channels = [cv2.imread(self.images_paths[k][1]) for k in indexes]
        
        #convert bgrtorgb
		images = [cv2.cvtColor(image,cv2.COLOR_BGR2RGB) for image in images]
		channels = [cv2.cvtColor(image,cv2.COLOR_BGR2RGB) for image in channels]
        
		# preprocess and augment data
		if self.augment == True:
			images,channels,labels = self.augmentor(images,channels, labels)
		

	
        #resize
		images = [cv2.resize(image, (self.dim[1],self.dim[0]) ) for image in images]
		channels = [cv2.resize(image, (self.dim[1],self.dim[0]) ) for image in channels]
		#combine
		images=[np.concatenate((image,channel),axis=2) for image,channel in zip(images,channels)]
		if self.IsVisualization:
				images = np.array(images)
		else:
				images = np.array([img for img in images])/255
		if self.ApplyMaskEnhancment:
			labels=[ApplyDialationAndBlurThresh(label) for label in labels]
		labels = [cv2.resize(label, (self.dim[1],self.dim[0]) )[:,:,0:1] for label in labels]
		labels = ((np.array(labels)>0)*255)/255
		return images, labels
	
	
	def augmentor(self, images,channels, labels):
		'Apply data augmentation. augment image and the regression label points.'
		augmentedImages=[]
		augmentedchannels=[]
		augmentedLabels=[]
		for image,channel,label in zip(images,channels,labels):
			det = self.seq.to_deterministic()
			img = det.augment_image(image)
			chn = det.augment_image(image)
			msk = det.augment_image(label.astype(np.uint8),
						hooks=ia.HooksImages(activator=hook))
			augmentedchannels.append(chn)
			augmentedImages.append(img)
			augmentedLabels.append(msk)
		return augmentedImages,augmentedchannels,augmentedLabels
    

def getTrainValDatasetGenarators(dir,image_dimensions = (256, 256, 6),train_size=0.8,random_state=4,batch_size=4,applyAug=False,IsVisualization=False,StartAt=None, Limit=None,ApplyMaskEnhancment=False):
    ##get list of available dataset
    filePaths=GetAvailableDatasetFiles(dir)
    if StartAt:
        filePaths=filePaths[StartAt:]
    if Limit:
        filePaths=filePaths[:Limit]
    #split dataset
    trainPaths,valPaths=sklearn.model_selection.train_test_split(filePaths,train_size=train_size,random_state=random_state)
    #load Dataset
    trainX,trainY=getXY(trainPaths)
    valX,valY=getXY(valPaths)
    #convert to numpy arrays
    trainX=np.array(trainX)
    trainY=np.array(trainY)
    valX=np.array(valX)
    valY=np.array(valY)
    if applyAug:

        trainGen=DataGenerator(trainX,trainY,batch_size=batch_size,image_dimensions=image_dimensions,augmentation=getAugmentor(),IsVisualization=IsVisualization,ApplyMaskEnhancment=ApplyMaskEnhancment)
    else:
        trainGen=DataGenerator(trainX,trainY,batch_size=batch_size,image_dimensions=image_dimensions,IsVisualization=IsVisualization,ApplyMaskEnhancment=ApplyMaskEnhancment)
    valGen=DataGenerator(valX,valY,batch_size=batch_size,image_dimensions=image_dimensions,IsVisualization=IsVisualization,ApplyMaskEnhancment=ApplyMaskEnhancment)

    return trainGen,valGen


def getTestDatasetGenarator(dir,image_dimensions = (256, 256, 6),test_size=0.2,random_state=4,batch_size=4,IsVisualization=False,ApplyMaskEnhancment=False):
    ##get list of available dataset
    filePaths=GetAvailableDatasetFiles(dir)
    #split dataset
    _,testPaths=sklearn.model_selection.train_test_split(filePaths,test_size=test_size,random_state=random_state)
    #load Dataset
    testX,testY=getXY(testPaths)
    testX=np.array(testX)
    testY=np.array(testY)
    testGen=DataGenerator(testX,testY,batch_size=batch_size,image_dimensions=image_dimensions,shuffle=False,IsVisualization=IsVisualization,ApplyMaskEnhancment=ApplyMaskEnhancment)

    return testGen

