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

class Controller:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/R1/pi_camera/image_raw', Image, self.image_callback)
        self.cmd_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        self.start_timer = 0
        self.count_license_plates = 0
        self.car_frame_counter = 0
        self.prev_car_frame = None


    # Detecting the red line ----------- WORKS very well
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img_car = cv_image[450:720, 300: 700]
            cv2.imshow("image", img_car)
            cv2.waitKey(1)
            # # plt.show()
        except CvBridgeError as e:
            print(e)

        if self.car_frame_counter % 3 == 0:
            if self.prev_car_frame is not None:
                # subtract the current frame from the previous frame
                diff = cv2.absdiff(img_car, self.prev_car_frame)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                
                # check for movement by counting the number of non-zero pixels
                if cv2.countNonZero(thresh) > 0:
                    print("Movement detected.")
                else:
                    self.movement_detected = False
                    print("No movement detected.")
                    
            # update the previous frame
            self.prev_car_frame = img_car.copy()
        
        # increment the counter
        self.car_frame_counter += 1




def main(args):
    rospy.init_node('Car', anonymous=True)
    lf = Controller()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
