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

model_path = '/home/fizzer/ros_ws/src/controller_pkg/node/modelGoodie.h5'
model = tf.keras.models.load_model(model_path)
model_path_left = '/home/fizzer/ros_ws/src/controller_pkg/node/modelGoodieLeft.h5'
modelLeft = tf.keras.models.load_model(model_path_left)

class Controller:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/R1/pi_camera/image_raw', Image, self.image_callback)
        self.cmd_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        self.start_timer = 0
        self.count_license_plates = 0
        # self.cnn_trainer = CNNTrainer((3,224,224) , 2)
        # self.model = tf.keras.models.load_model(model_path, custom_objects={'bc_loss': self.cnn_trainer.build_model}) # Load the saved model
        # self.model = tf.keras.models.load_model(model_path)
        self.countTerrain = 0
        # self.model = tf.keras.models.load_model(model_path)
        self.start_timer = time.time()
        self.is_turning = False



    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            img = cv_image[400:720, 640: 1280]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

            cv2.imshow("Image", img)
            cv2.waitKey(1)
            # # plt.show()
        except CvBridgeError as e:
            print(e)

        try:
            # Pass the image through the model
            # angular_vel = float(self.telemetry(cv_image))
            angular_vel = float(self.telemetryLeft(cv_image, modelLeft))
        except Exception as e:
            print("Error predicting: ", str(e))
        
        linear_vel = 0.00

        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel * 2.7
        self.cmd_pub.publish(twist)

        self.start_timer += TIME_STEP
        print(self.start_timer)

    def preProcessing(self,img):
        img = img[400:720, 640: 1280]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img,(3,3),0)
        img = cv2.resize(img, (200,66))
        img = img / 255
        return img
    
    def telemetry(self,img):
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
