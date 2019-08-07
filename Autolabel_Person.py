######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import PIL.Image as Image
from Create_Xml import Xml
from lxml import etree
from Extract_parameter import Extract
import shutil
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'


# Grab path to current working directory
CWD_PATH = os.getcwd()
CW_DIR_PATH=os.path.split(CWD_PATH)[1]

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE_DIR = os.path.join(CWD_PATH,'images','test')


# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')




# read all image
dir=os.listdir(PATH_TO_IMAGE_DIR)
for filename in dir :
    if str.lower(os.path.splitext(filename)[1])=='.jpg' :
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        PATH_TO_IMAGE = os.path.join(PATH_TO_IMAGE_DIR, filename)
        print(filename)
        image = cv2.imread(PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Extract parameter from the detection result
        extract=Extract()
        all_box_and_classname = Extract.extract_parameter(np.squeeze(boxes),
                                                  np.squeeze(scores),
                                                  np.squeeze(classes).astype(np.int32),
                                                  category_index        )

        #Determine whether an object has been detected
        if not all_box_and_classname.items() :
            print(filename+" is not detected")
            if  not os.path.isdir(PATH_TO_IMAGE_DIR + '\\'+'ND_data'):
                os.makedirs(PATH_TO_IMAGE_DIR + '\\'+'ND_data')
            shutil.move(PATH_TO_IMAGE_DIR+'\\'+filename, PATH_TO_IMAGE_DIR + '\\' + 'ND_data')
        else:
            #Get image width , height
            image_pil = Image.fromarray(image)
            im_width, im_height = image_pil.size

            #Create xml and save  label messages
            item=Xml()
            item_CDM=item.create_dir_message(item.annotation,CW_DIR_PATH,filename,PATH_TO_IMAGE)
            item_imgSize=item.image_size(item_CDM,im_width,im_height)
            count=0
            for box, class_name in all_box_and_classname.items():
                ymin, xmin, ymax, xmax  = box
                (xmin, xmax, ymin, ymax) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)
                if count==0:
                    item_Object=item.create_object(item_imgSize,class_name,xmin, xmax, ymin, ymax)
                    count+=1
                else :
                    item_Object = item.create_object(item_Object, class_name, xmin, xmax, ymin, ymax)

            result = etree.ElementTree(item_Object)

            result.write(PATH_TO_IMAGE_DIR+'//'+filename.split('.')[0]+'.xml')
        print('----next picture---')















