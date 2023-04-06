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
        self.start_timer = 0
        self.count_license_plates = 0
        # self.cnn_trainer = CNNTrainer((3,224,224) , 2)
        # self.model = tf.keras.models.load_model(model_path, custom_objects={'bc_loss': self.cnn_trainer.build_model}) # Load the saved model
        # self.model = tf.keras.models.load_model(model_path)
        self.countTerrain = 0
        # self.model = tf.keras.models.load_model(model_path)
        self.start_timer = 0
        self.is_turning = False
        self.prev_frame = None  # to store the previous frame
        self.frame_counter = 0  # to count the frames

        self.pid = PID(2.6, 0.20,RIGHT_MOST_X_COORDINATE) # KP = 2, KD = 0.5, and x speed = 0.25 below was rlly good
        self.pidLeft = PID(3, 0.24, LEFT_MOST_X_COORDINATE)


    # # Detecting the red line ----------- WORKS very well
    # def image_callback(self, msg):
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    #         img_red = cv_image[600:720, 300: 1100]

    #         # # plt.show()
    #     except CvBridgeError as e:
    #         print(e)
    #     # Convert the image to HSV color space
    #     hsv_img_red = cv2.cvtColor(img_red, cv2.COLOR_BGR2HSV)

    #     # Define the range of red color in HSV color space
    #     lower_red = np.array([0, 50, 50])
    #     upper_red = np.array([10, 255, 255])

    #     # Create a mask using the range of red color
    #     mask1_red = cv2.inRange(hsv_img_red, lower_red, upper_red)

    #     # Define the range of red color in HSV color space
    #     lower_red = np.array([170, 50, 50])
    #     upper_red = np.array([180, 255, 255])

    #     # Create another mask using the range of red color
    #     mask2_red = cv2.inRange(hsv_img_red, lower_red, upper_red)

    #     # Combine the masks
    #     mask_red = mask1_red + mask2_red

    #     # Apply the mask to the original image
    #     red_img = cv2.bitwise_and(img_red, img_red, mask=mask_red)

    #     gray_red = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)

    #     ##remove noise using gaussian blur 
    #     blured_image_red = cv2.GaussianBlur(gray_red, (3,3), 0)

    #     ##detect edges from luminosity change 
    #     edges_red = cv2.Canny(blured_image_red, 100, 200)

    #     red_line_points = np.count_nonzero(edges_red)

    #     print(red_line_points)

    #     # Display the red image
    #     cv2.imshow("Red Image", edges_red)
    #     cv2.waitKey(1)

    # Pedestrian Detection ----------------------------
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img_ped = cv_image[250:500, 550:750]
            cv2.imshow("Image", img_ped)
            cv2.waitKey(1)

            # # plt.show()
        except CvBridgeError as e:
            print(e)



    def region(self,image):
        height, width = image.shape

        polygon = np.array([[(int(width), height), (int(width),  
        int(height*0.90)), (int(width*0.00), int(height*0.90)), (int(0), height),]], np.int32)    

        mask = np.zeros_like(image)

        mask = cv2.fillPoly(mask, polygon, 255)
        mask = cv2.bitwise_and(image, mask)

        return mask
    
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
    rospy.init_node('Pedestrian', anonymous=True)
    lf = Controller()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
