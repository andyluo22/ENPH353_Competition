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

model_path = '/home/fizzer/ros_ws/src/controller_pkg/node/modelBestNew2.h5'
model_path2 = '/home/fizzer/ros_ws/src/controller_pkg/node/modelBestNew.h5'
model = tf.keras.models.load_model(model_path)
model2 = tf.keras.models.load_model(model_path2)

class Controller:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/R1/pi_camera/image_raw', Image, self.image_callback)
        self.cmd_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        self.start_timer = 0
        self.count_license_plates = 0
        self.countTerrain = 0
        self.mode = "stop_and_turn"
        self.last_action_time = 0

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            img = cv_image[400:720,:,:]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

            cv2.imshow("Image", img)
            cv2.waitKey(1)
            # # plt.show()
        except CvBridgeError as e:
            print(e)

        if self.countTerrain < 1.4:
            linear_vel = 0.20
            twist = Twist()
            twist.linear.x = linear_vel 
            twist.angular.z = 0.05
            self.cmd_pub.publish(twist)
        elif self.countTerrain >= 1.4 and self.countTerrain < 3.8:
            if self.mode == "stop_and_turn":
                if self.start_timer - self.last_action_time < 1.4:
                    try:
                        # Pass the image through the model
                        angular_acc = float(self.telemetry(cv_image,model))
                    except Exception as e:
                        print("Error predicting: ", str(e))

                    linear_vel = 0.00
                    twist = Twist()
                    twist.linear.x = linear_vel
                    twist.angular.z = angular_acc * 2
                    self.cmd_pub.publish(twist)
                else:
                    # Switch to "go_straight" mode
                    self.mode = "go_straight"
                    self.last_action_time = self.start_timer
            
            elif self.mode == "go_straight":
                if self.start_timer - self.last_action_time < 0.5:
                    # Use PID control to generate Twist message with zero angular acceleration
                    linear_vel = 0.20
                    twist = Twist()
                    twist.linear.x = linear_vel
                    twist.angular.z = 0.0
                    self.cmd_pub.publish(twist)
                else:
                    # Switch to "stop_and_turn" mode
                    self.mode = "stop_and_turn"
                    self.last_action_time = self.start_timer
            
            self.start_timer += TIME_STEP
            print(self.start_timer)
        else:
            if self.mode == "stop_and_turn":
                if self.start_timer - self.last_action_time < 1.4:
                    # Use PID control to generate Twist message with angular acceleration from the model
                    try:
                        # Pass the image through the model
                        angular_acc = float(self.telemetry(cv_image, model2))
                    except Exception as e:
                        print("Error predicting: ", str(e))

                    linear_vel = 0.00
                    twist = Twist()
                    twist.linear.x = linear_vel
                    twist.angular.z = angular_acc * 2
                    self.cmd_pub.publish(twist)
                else:
                    # Switch to "go_straight" mode
                    self.mode = "go_straight"
                    self.last_action_time = self.start_timer
            
            elif self.mode == "go_straight":
                if self.start_timer - self.last_action_time < 0.5:
                    # Use PID control to generate Twist message with zero angular acceleration
                    linear_vel = 0.20
                    twist = Twist()
                    twist.linear.x = linear_vel
                    twist.angular.z = 0.0
                    self.cmd_pub.publish(twist)
                else:
                    # Switch to "stop_and_turn" mode
                    self.mode = "stop_and_turn"
                    self.last_action_time = self.start_timer
            
            self.start_timer += TIME_STEP
            print(self.start_timer)
        self.countTerrain += TIME_STEP

        

    def preProcessing(self,img):
        img = img[400:720, :, :]
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