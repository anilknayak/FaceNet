# FaceNet
Face Recognition 

## Library Used:
1. <b>Dlib</b> : To detect face and prepare face images for training
2. imutils : for utility purpose
3. cv2 : camera and image read & write, image bluring
4. tensorflow : prepare and training network for face recognition

## How to Use this Lib
### 1. Have your all training images in like below folder structure
   - base_dir <facenet>
      - images
        - data
          - class 1
              - image 1
              - image 2
          - class 2
              - image 1
              - image 2
  
  #### Note: 
  1. All the above example as folder named as class 1 and class 2 folders, defines the class label for each training data
  2. Images could be a single faces of 180x180 size or could be a large image i.e. 600x1200 having single or multiple face
 
 ### 2. Edit face_recognition.config file for your training
1. First of all, if you have large image then your need to have pre processing to identify the faces in the image and prepare the class for training, as you have already prepared your class label as folder name and copied all class specific images inside.
 then update "pre_processing_required":true option to true so that it will prepare the training face images of 180x180 or you have change the side of the image you have to prepare in 
 "image" :{
    "resize": {
      "width":180,
      "height":180,
      "required": false
    },
    "width":180,
    "height":180
  }
  
2. if you have already have face images of 180x180 or 28x28 or 36x36 or 200x200 for training then you can make the option
  "pre_processing_required":false
  
3.  Change the following properties for folder training details
"training":{
    "base_directory":"/FaceNet/",   ### Base Folder of Api
    "image_directory" : "images/",  ### Image folder in side base_directory
    "training_data_folder" : "images/train/", 
    "testing_data_folder" : "images/test/",
    "structure_of_data": "folder",
    "random_shuffle":true,  ### if you want to have random shuffle of data for training and testing
    "training_size_percentage":95, ### splitting of data into training 95% and testing 5%
    "training_steps":100, ### number of epoch for training 
    "batch_size":24, ### Batch side for training
    "learning_rate":0.001 ### initial learning rate
  },
  
 4. Then change tje 
 
  "network_config":{
    "tensor_name":"auto", ### auto generated tensor name
    "input_size":"auto", ### input layer side auto decided as per the image structrue mentioned as above 180x180
    "output_size":"auto", ### output size will be decided on number of class folders are there in image/data directory as mentioned above
    "network":[...] ### Heart of the network lies here. Check the sample configuration file and check the network configuration and play with it more as you understand 
    
### Outcome
As I have trainined with 460 faces of 28 classes, i have got 68% of accuracy 
    }


