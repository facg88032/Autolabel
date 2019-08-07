Autolabel with Tensorflow objcet detection
===

## Introduction
In order to alleviate label tens of thousands of images by hand that spends much time and vigor, so I development AutoLabel.<br>

The core idea of this project is to use the pre-train model to detection unlabeled images ,  and record the detection results (bounding box, class, filename, etc) as text (ex: xml). 
I choice [LabelImg](https://github.com/tzutalin/labelImg) to check and fine-tune the result because labelImg has visualization UI.
In addition, if you use different models in the future, you transfer text formats thar models need.

## AutoLabel
### train model
Pre-train models of this time chooses [Tensorflow Objection Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) , I use 1000 manaul label images to train.
How to train the model you can reference [TensorFlow-Object-Detection-API-Tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10).<br>

### object detection on images
When training is finished , we can start to detection the image. The AutoLabel_perspn.py code is use object_detection_image.py of [TensorFlow-Object-Detection-API-Tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) to modify.

#### Ex:
detction on images<br>
![image](https://github.com/facg88032/picture/blob/master/image_person1.png)<br>

### record the result
Because [LabelImg](https://github.com/tzutalin/labelImg) use Xml format , so I choose xml format to record the result. (Create_xml.py)<br>
you can record the parameter that what you need like filename , class , bounding box , imagesize ,etc.
#### Ex:
labelImg of xml format<br>
![image](https://github.com/facg88032/picture/blob/master/labelImage_xml_format.png)

### LabelImg check and fine-tune
LabelImg read a.xml file containing label data fo each image,so we can check the detection result.<br>
If results have some error , we can fine-tune on LabelImg.
#### Ex:
read image and xml on labelImg
![image](https://github.com/facg88032/picture/blob/master/labelImage.png)

### End
Now Autolabel is finished , you can input new data to train your models that makes better performance. 
