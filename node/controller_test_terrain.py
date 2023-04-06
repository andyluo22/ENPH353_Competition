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
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String


CAM_IMAGE_HEIGHT = 720
CAM_IMAGE_WIDTH = 1280
TIME_STEP = 0.040
PID_SCALE = 250
SCALE_SPEED = 10
LANE_WIDTH = 80 # about 1/13 of 1280 so i just used 80 which seems like a good give or take
RIGHT_MOST_X_COORDINATE = 1220
LEFT_MOST_X_COORDINATE = 60

model_path = '/home/fizzer/ros_ws/src/controller_pkg/node/modelGoodie.h5'
model = tf.keras.models.load_model(model_path)


class Controller:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/R1/pi_camera/image_raw', Image, self.image_callback)
        self.cmd_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        self.pid = PID(2.6, 0.20,RIGHT_MOST_X_COORDINATE) # KP = 2, KD = 0.5, and x speed = 0.25 below was rlly good
        self.pidLeft = PID(3, 0.24, LEFT_MOST_X_COORDINATE)
        self.start_timer = 0
        self.terrain_timer = 0
        self.count_license_plates = 0
        self.countTerrain = 0

        self.command_timer = time.time()
        self.terrain_timer = 0
        self.is_turning = False
        self.time_terrain_tracker = 0
        self.movement_detected = True
        self.count_red_lines = 0
        self.prev_frame = None  # to store the previous frame
        self.frame_counter = 0  # to count the frames


        self.off_terrain = False





    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        except CvBridgeError as e:
            print(e)
        try:
            # Pass the image through the model
            angular_vel = float(self.telemetry(cv_image, model))
        except Exception as e:
            print("Error predicting: ", str(e))

        if self.time_terrain_tracker < 0.95:
            linear_vel = 0.1925

        elif self.time_terrain_tracker >= 0.95 and self.time_terrain_tracker < 23:
            linear_vel = 0.07
            angular_vel = angular_vel * 2.6

        else:
            linear_vel = 0
        



        
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel 
        self.cmd_pub.publish(twist)
        self.time_terrain_tracker += TIME_STEP
        print("Terrain Timer\n")
        print(self.time_terrain_tracker)
        print("\n")
        
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

    def preProcessing(self,img):
        img = img[400:720, 640:1280]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img,(3,3),0)
        img = cv2.resize(img, (200,66))
        img = img / 255
        return img
    
    def telemetry(self,img, model):
        image = np.asarray(img)
        image = self.preProcessing(image)
        image = np.array([image])
        angular_z = float(model.predict(image))
        print(angular_z)
        return angular_z
    
    def preProcessingLeft(self,img):
        img = img[400:720, 0:640]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img,(3,3),0)
        img = cv2.resize(img, (200,66))
        img = img / 255
        return img
    
    def telemetryLeft(self,img, model):
        image = np.asarray(img)
        image = self.preProcessingLeft(image)
        image = np.array([image])
        angular_z = float(model.predict(image))
        print(angular_z)
        return angular_z
    


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
