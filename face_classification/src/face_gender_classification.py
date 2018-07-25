#!/usr/bin/env python
import rospy
import sys

# Ros Messages
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

# We do not use cv_bridge it does not support CompressedImage in python
from cv_bridge import CvBridge, CvBridgeError
from statistics import mode
import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

import tensorflow as tf


USE_LOCAL_CAMERA=False

class EmotionGenderListener:

	def __init__(self):

            # Initialize the node with rosp
            rospy.init_node('emotion_recognizer_node', anonymous=True)
            
            rospy.loginfo("recognizer started")
            print "1................................................"
            
            if(USE_LOCAL_CAMERA):
                self.video_capture = cv2.VideoCapture(0)
                            
            self._detection_models = "~detection_models"
            if rospy.has_param(self._detection_models):
                self.detection_model_path = rospy.get_param(self._detection_models)
            else:
                rospy.logwarn("parameters need to be set to start recognizer.")
                return
            
            
            self.emotion_models = "~emotion_models"
            if rospy.has_param(self.emotion_models):
                self.emotion_model_path = rospy.get_param(self.emotion_models)
            else:
                rospy.logwarn("parameters need to be set to start recognizer.")
                return
            
            
            self.gender_models = "~gender_models"
            if rospy.has_param(self.gender_models):
                self.gender_model_path = rospy.get_param(self.gender_models)
            else:
                rospy.logwarn("parameters need to be set to start recognizer.")
                return
        
        
            self.bridge = CvBridge()
    
        # parameters for loading data and images
            self.emotion_labels = get_labels('fer2013')
            self.gender_labels = get_labels('imdb')

        # hyper-parameters for bounding boxes shape
            self.frame_window = 10
            self.emotion_offsets = (20, 40)
            self.gender_offsets = (30, 60)

        # loading models
            self.face_detection = load_detection_model(self.detection_model_path)
            self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
            self.gender_classifier = load_model(self.gender_model_path, compile=False)

        # getting input model shapes for inference
            self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
            self.gender_target_size = self.gender_classifier.input_shape[1:3]

        # starting lists for calculating modes
            self.emotion_window = []
            self.emotion_publisher = rospy.Publisher("/qt_face/setEmotion",String,queue_size=10)
            self.gender_window = []
            self.gender_publisher = rospy.Publisher("/gender",String,queue_size=10)

        # self.speech_publisher = rospy.Publisher("/speaker",String,queue_size=10)
            self.emotion_msg = String()
            self.gender_msg = String()
            self.speech_msg = String()

            #Where to publish
            self._output_image_topic = "~image_topic_output"
            print rospy.has_param(self._output_image_topic)
            if rospy.has_param(self._output_image_topic):
                output_image_topic = rospy.get_param(self._output_image_topic)
                self.image_pub = rospy.Publisher(output_image_topic,Image, queue_size=10)
        
            # Scaling factor for face recognition image
            self.scaling_factor = 0.50
            
            #Where to subscribe
            self._input_image_topic = "~image_topic_input"
            print rospy.has_param(self._input_image_topic)
            if rospy.has_param(self._input_image_topic):
                input_image_topic = rospy.get_param(self._input_image_topic)
                if(not USE_LOCAL_CAMERA):
                    self.image_sub = rospy.Subscriber(input_image_topic, Image, self.callback)
                    
            self.graph = tf.get_default_graph()
                
                

    #if you want to use the camera, uncomment next line, comment the callback, and set USE_LOCAL_CAMERA to True
    #while True:#### direct conversion to CV2 ####
        def callback(self,data):
            #/////////////////////////////////////////////
            #cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
            #cv_image = cv2.resize(cv_image, target_size)  # resize image
            #np_image = np.asarray(cv_image)               # read as np array
            #np_image = np.expand_dims(np_image, axis=0)   # Add another dimension for tensorflow
            #np_image = np_image.astype(float)  # preprocess needs float64 and img is uint8
            #np_image = preprocess_input(np_image)         # Regularize the data
            #/////////////////////////////////////////////
            if(not USE_LOCAL_CAMERA):                                                    
                try:
                        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
                except CvBridgeError as e:
                        print(e)

            # Capture frame-by-frame
            if(USE_LOCAL_CAMERA):
                ret, frame1 = self.video_capture.read()
            #/////////////////////////////////////////////    
            
            
            
            #print"--------"
            #print('input_msg height  : {}'.format(frame.height))
            #print('input_msg width   : {}'.format(frame.width))
            #print('input_msg step    : {}'.format(frame.step))
            #print('input_msg encoding: {}'.format(frame.encoding))
            #print('output dtype      : {}'.format(frame.dtype))
            #print('output shape      : {}'.format(frame.shape))
            #print"--------"
            
            
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detect_faces(self.face_detection, gray_image)

            for face_coordinates in faces:

                    x1, x2, y1, y2 = apply_offsets(face_coordinates, self.gender_offsets)
                    rgb_face = rgb_image[y1:y2, x1:x2]

                    x1, x2, y1, y2 = apply_offsets(face_coordinates, self.emotion_offsets)
                    gray_face = gray_image[y1:y2, x1:x2]

                    try:
                            gray_face = cv2.resize(gray_face, (self.emotion_target_size))
                            rgb_face = cv2.resize(rgb_face, (self.gender_target_size))
                    except:
                            continue

                    gray_face = preprocess_input(gray_face, True)
                    gray_face = np.expand_dims(gray_face, 0)
                    gray_face = np.expand_dims(gray_face, -1)

                    rgb_face = np.expand_dims(rgb_face, 0)
                    rgb_face = preprocess_input(rgb_face, False)


                    ## This is a workaround for asynchronous execution using TF and ROS
                    # https://github.com/keras-team/keras/issues/2397
                    # http://projectsfromtech.blogspot.com/2017/10/visual-object-recognition-in-ros-using.html
                    #global self.graph                                  
                    with self.graph.as_default():

                        emotion_prediction = self.emotion_classifier.predict(gray_face)
                        emotion_probability = np.max(emotion_prediction)
                        emotion_label_arg = np.argmax(emotion_prediction)
                        emotion_text = self.emotion_labels[emotion_label_arg]

                        gender_prediction = self.gender_classifier.predict(rgb_face)
                        gender_probability = np.max(gender_prediction)
                        gender_label_arg = np.argmax(gender_prediction)
                        gender_text = self.gender_labels[gender_label_arg]


                        print emotion_text
                        print(emotion_probability)
                        print('%')
                        self.emotion_window.append(emotion_text)

                        print gender_text
                        print(gender_probability)
                        print('%')
                        self.gender_window.append(gender_text)

        
                        self.emotion_msg.data = 'ava_' + emotion_text
                        self.emotion_publisher.publish(self.emotion_msg)
                        self.gender_msg.data = 'ava_' + gender_text
                        self.gender_publisher.publish(self.gender_msg)
                        #self.speech_msg.data = 'I see that you are ' + emotion_text
                        #self.speech_publisher.publish(speech_msg)

                        if len(self.emotion_window) > self.frame_window:
                            self.emotion_window.pop(0)
                            self.gender_window.pop(0)
                        try:
                            emotion_mode = mode(self.emotion_window)
                            gender_mode = mode(self.gender_window)
                        except:
                            continue

                        if emotion_text == 'angry':
                            color = emotion_probability * np.asarray((255, 0, 0))
                        elif emotion_text == 'sad':
                            color = emotion_probability * np.asarray((0, 0, 255))
                        elif emotion_text == 'happy':
                            color = emotion_probability * np.asarray((255, 255, 0))
                        elif emotion_text == 'surprise':
                            color = emotion_probability * np.asarray((0, 255, 255))
                        elif emotion_text == 'disgust':
                            color = emotion_probability * np.asarray((255, 0, 255))
                        elif emotion_text == 'fear':
                            color = emotion_probability * np.asarray((255, 255, 255))
                        else:
                            color = emotion_probability * np.asarray((0, 255, 0))

                        color = color.astype(int)
                        color = color.tolist()

                        draw_bounding_box(face_coordinates, rgb_image, color)
                        draw_text(face_coordinates, rgb_image, emotion_mode,
                                color, 0, -45, 1, 1)
                        draw_text(face_coordinates, rgb_image, gender_mode,
                                color, 0, -20, 1, 1)
                    
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(rgb_image, "rgb8"))
                # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                # cv2.imshow('window_frame', bgr_image)
            except CvBridgeError as e:
                print(e)

            #cv2.imshow("Image window", rgb_image)
            #cv2.waitKey(3)

        #try:
            #self.image_pub.publish(self.bridge.cv2_to_imgmsg(bgr_image, "rgb8"))
        #except CvBridgeError as e:
            #print(e)



if __name__ == '__main__':
  rospy.loginfo("Simple Emotion Detection ...........")
  print "................................................"
  ic = EmotionGenderListener()
#   cv2.namedWindow('window_frame')
  
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()
 
