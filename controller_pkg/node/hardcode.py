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

model_path = '/home/fizzer/ros_ws/src/controller_pkg/node/my_model.pt5'


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
        self.model = tf.keras.models.load_model(model_path) # Load the saved model
        self.countTerrain = 0


    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_imageResize = cv2.resize(cv_image, (224, 224)) # resize the image to 224x224
            input_data = np.transpose(cv_imageResize, (2, 0, 1))  # transpose to (3, 224, 224)
            input_data = input_data.astype(np.float32) / 255.0  # scale pixel values to be between 0 and 1

            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # plt.show()
        except CvBridgeError as e:
            print(e)

        state = np.zeros(10, dtype=np.int)

        copy = np.copy(cv_image)

        # ----------------------------NON-HSV
        gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)

        blured_image = cv2.GaussianBlur(gray, (3,3), 0)

        edges = cv2.Canny(blured_image, 0, 50)

        # cv2.imshow("edges", edges)
        # cv2.waitKey(1)

        isolated = self.region(edges)

        # cv2.imshow("isolated", isolated)
        # cv2.waitKey(1)

        row_indexes, col_indexes = np.nonzero(isolated)
        max_col = 0
        min_col = 0

        # Find find left side of edge and right side of edge
        if len(col_indexes) > 0:
            max_col = max(col_indexes)
            min_col = min(col_indexes)

        cv2.imshow("raw", cv_image)
        cv2.waitKey(1)

        # ----------------------------HSV

        lower_green = np.array([10, 70, 100])
        upper_green = np.array([35, 160, 200])

        # Create mask for yellow-green grass
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Define color thresholds for white lanes in HSV
        lower_white = np.array([130, 0, 200])
        upper_white = np.array([255, 100, 255])

        # Create mask for white lanes
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # cv2.imshow("white" , white_mask)
        # cv2.waitKey(1)

        # Combine masks using bitwise_and()
        mask = cv2.bitwise_or(green_mask, white_mask)

        # Close any gaps in the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow("raw", mask)
        # cv2.waitKey(1)

        # Apply mask to original image
        masked_img = cv2.bitwise_and(cv_image, cv_image, mask=mask)

        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        ##remove noise using gaussian blur 
        blured_image = cv2.GaussianBlur(gray, (3,3), 0)

        ##detect edges from luminosity change 
        edges = cv2.Canny(blured_image, 100, 200)

        ##mask polygon onto black plane 
        isolated = self.region(edges)

        # Detect lines using the Hough transform
        lines = cv2.HoughLinesP(isolated, 1, np.pi/180, 50, minLineLength=20, maxLineGap=10)

        max_x = 0

        # HOUGH LINES
        if(lines is not None):
            for line in lines:
                x1, y1, x2, y2 = line[0]

                start_x = min(x1,x2)
                end_x = max(x1,x2)

                if end_x > max_x:
                    max_x = end_x

                cv2.line(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # self.countTerrain = self.countTerrain + 1
            

        cv2.imshow("raw", cv_image)
        cv2.waitKey(1)

        # --------------------------------------- MOVEMENT CONTROL
        if self.start_timer < 5 and lines is None:
            twist = Twist()
            twist.linear.x = 0.14
            if(min_col > 1000):
                min_col = 50
            twist.angular.z = self.pidLeft.computeLeft(min_col)
            self.cmd_pub.publish(twist)

            self.start_timer += TIME_STEP
            print(self.start_timer)

        elif lines is None and self.countTerrain == 0: 
            twist = Twist() 
            twist.linear.x = 0.30
            twist.angular.z = self.pid.computeRight(max_col)
            self.cmd_pub.publish(twist)

            self.start_timer += TIME_STEP
            print(self.start_timer)

        elif lines is not None and self.countTerrain == 0:
            twist = Twist() 
            twist.linear.x = 0
            twist.angular.z = 0
            self.cmd_pub.publish(twist)
            self.countTerrain = self.countTerrain + 1
            time.sleep(1)
            # twist.linear.x = 0.12
            # if(max_x > 900):
            #     twist.angular.z = self.pid.computeRight(max_x)

            # self.cmd_pub.publish(twist)
            # self.start_timer += TIME_STEP
            # print(self.start_timer)
        elif self.countTerrain == 1:
            twist = Twist() 
            twist.linear.x = 0.25
            twist.angular.z = 0
            self.cmd_pub.publish(twist)
            time.sleep(1)
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            time.sleep(0.5)
            self.countTerrain = self.countTerrain + 1
            print("time done")
            

        elif self.countTerrain == 2:
            twist = Twist() 
            twist.linear.x = 0.0
            twist.angular.z = 0.3
            self.cmd_pub.publish(twist)
            time.sleep(2)
            twist.linear.x = 0.25
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            time.sleep(0.8)
            twist.linear.x = 0.0
            twist.angular.z = 0.83
            self.cmd_pub.publish(twist)
            time.sleep(1.80)
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            time.sleep(0.5)
            self.countTerrain = self.countTerrain + 1
            print("time done")
            
        elif self.countTerrain == 3:
            twist = Twist() 
            twist.linear.x = 0.5
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            time.sleep(3)
            self.countTerrain = self.countTerrain + 1

        else:
            twist = Twist() 
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)





    
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