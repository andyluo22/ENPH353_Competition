#! /usr/bin/env python3

import rospy
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import sys
import tensorflow as tf
import time

from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend

from cnn_trainer import CNNTrainer

CAM_IMAGE_HEIGHT = 720
CAM_IMAGE_WIDTH = 1280
TIME_STEP = 0.040
PID_SCALE = 250
SCALE_SPEED = 10
LANE_WIDTH = 80 # about 1/13 of 1280 so i just used 80 which seems like a good give or take
RIGHT_MOST_X_COORDINATE = 1220
LEFT_MOST_X_COORDINATE = 60

model_path = '/home/fizzer/ros_ws/src/controller_pkg/node/imitation_model.h5'
cnn_trainer = CNNTrainer((3,224,224) , 2)
model = tf.keras.models.load_model(model_path)

class Controller:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/R1/pi_camera/image_raw', Image, self.image_callback)
        self.cmd_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        self.pid = PID(2.6, 0.25,RIGHT_MOST_X_COORDINATE) # KP = 2, KD = 0.5, and x speed = 0.25 below was rlly good
        self.pidLeft = PID(3, 0.24, LEFT_MOST_X_COORDINATE)
        self.start_timer = 0
        self.count_license_plates = 0
        # self.cnn_trainer = CNNTrainer((3,224,224) , 2)
        # self.model = tf.keras.models.load_model(model_path, custom_objects={'bc_loss': self.cnn_trainer.build_model}) # Load the saved model
        # self.model = tf.keras.models.load_model(model_path)
        self.countTerrain = 0
        # self.model = tf.keras.models.load_model(model_path)


    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_imageResize = cv2.resize(cv_image, (224, 224)) # resize the image to 224x224
            input_data = np.transpose(cv_imageResize, (2, 0, 1))  # transpose to (3, 224, 224)
            input_data = input_data.astype(np.float32) / 255.0  # scale pixel values to be between 0 and 1
            img_tensor = tf.expand_dims(input_data, axis = 0)

            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # plt.show()
        except CvBridgeError as e:
            print(e)

        try:
            # Pass the image through the model
            predictions = model.predict(img_tensor)

        except Exception as e:
            print("Error predicting: ", str(e))


        linear_vel, angular_vel = predictions[0]

        # Use PID control to generate Twist message
        twist = Twist()
        twist.linear.x = linear_vel 
        twist.angular.z = angular_vel 
        print("Predictions")
        print(twist.linear.x)
        print(twist.angular.z)
        self.cmd_pub.publish(twist)


        self.start_timer += TIME_STEP
        print(self.start_timer)
    

    def region(self,image):
        height, width = image.shape

        polygon = np.array([[(int(width), height), (int(width),  
        int(height*0.90)), (int(width*0.00), int(height*0.90)), (int(0), height),]], np.int32)    

        mask = np.zeros_like(image)

        mask = cv2.fillPoly(mask, polygon, 255)
        mask = cv2.bitwise_and(image, mask)

        return mask
    
    def arrayOfPosition (self, width, x_coordinate, state):
        newWidth = np.divide(width,10)

        if(x_coordinate <= newWidth):
            state[0] = 1
        elif (x_coordinate <= newWidth * 2) :
            state[1] = 1
        elif (x_coordinate <= newWidth * 3) :
            state[2] = 1
        elif (x_coordinate <= newWidth * 4) :
            state[3] = 1
        elif (x_coordinate <= newWidth * 5) :
            state[4] = 1
        elif (x_coordinate <= newWidth * 6) :
            state[5] = 1
        elif (x_coordinate <= newWidth * 7) :
            state[6] = 1
        elif (x_coordinate <= newWidth * 8) :
            state[7] = 1
        elif (x_coordinate <= newWidth * 9) :
            state[8] = 1
        elif (x_coordinate <= newWidth * 10) :
            state[9] = 1


class PID:
    ##Initialize/construct 
    def __init__(self,KP,KD,target):
        self.kp = KP
        self.kd = KD
        self.setpoint = target
        self.error = 0
        self.error_last = 0
        self.derivative_error = 0.15
        self.output = 0
    def computeLeft(self, x_coordinate):
        self.error = self.setpoint - x_coordinate
        self.derivative_error = (self.error- self.error_last) / TIME_STEP
        self.error_last = self.error
        self.output = self.kp*self.error + self.kd*self.derivative_error
        self.output_scaled = np.divide(self.output,PID_SCALE)
        self.output_scaled = self.output_scaled
        return self.output_scaled

    def computeRight(self, x_coordinate):
        self.error = self.setpoint - x_coordinate
        self.derivative_error = (self.error- self.error_last) / TIME_STEP
        self.error_last = self.error
        self.output = self.kp*self.error + self.kd*self.derivative_error
        self.output_scaled = np.divide(self.output,PID_SCALE)
        self.output_scaled = self.output_scaled
        return self.output_scaled

def bc_loss(y_true, y_pred):
    # Compute the difference between predicted and expert actions
    error = y_pred - y_true

    # Compute the mean squared error
    mse = tf.reduce_mean(tf.square(error))

    # Scale the error by the norm of the expert actions
    norm = tf.norm(y_true)
    scaled_error = mse / (norm + 1e-8)

    return scaled_error

def main(args):
    rospy.init_node('Controller', anonymous=True)
    lf = Controller()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
    main(sys.argv)