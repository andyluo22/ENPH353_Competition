#! /usr/bin/env python3

import rospy
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import sys

CAM_IMAGE_HEIGHT = 480
CAM_IMAGE_WIDTH = 640
TIME_STEP = 0.5
PID_SCALE = 250
LANE_WIDTH = 80 # about 1/13 of 1280 so i just used 80 which seems like a good give or take


class Controller:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/R1/pi_camera/image_raw', Image, self.image_callback)
        self.cmd_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        self.pid = PID(14,2,CAM_IMAGE_WIDTH/2)


    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            # h, s, v = cv2.split(hsv_image)
            # print("Hue\n")
            # print(str(h) + "\n")
            # print("Saturation\n")
            # print(str(s) + "\n")
            # print("Value\n")
            # print(str(v) + "\n")
            # cv2.imshow("raw", cv_image)
            # cv2.waitKey(1)
            # Extract region of interest
            # roi = hsv_image[590:720, 0:1240, :]
            
            # # Calculate mean of hue, saturation, and value channels of ROI
            # mean_h = np.mean(roi[:,:,0])
            # mean_s = np.mean(roi[:,:,1])
            # mean_v = np.mean(roi[:,:,2])
            
            # print("Mean Hue:", mean_h)
            # print("Mean Saturation:", mean_s)
            # print("Mean Value:", mean_v)
            
            # plt.show()
        except CvBridgeError as e:
            print(e)

        # Define color thresholds for yellow-green grass in HSV
        # lower_green = np.array([20, 90, 150])
        # upper_green = np.array([35, 110, 170])

        # # Create mask for yellow-green grass
        # green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # # Define color thresholds for white lanes in HSV
        # lower_white = np.array([100, 0, 200])
        # upper_white = np.array([255, 130, 255])

        # # -------------------------------------- Seems to work need to adjust thresholds
        # # Define color thresholds for yellow-green grass in HSV
        # lower_green = np.array([10, 70, 100])
        # upper_green = np.array([35, 160, 200])

        lower_green = np.array([10, 70, 100])
        upper_green = np.array([35, 160, 200])

        # Create mask for yellow-green grass
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Define color thresholds for white lanes in HSV
        lower_white = np.array([130, 0, 200])
        upper_white = np.array([255, 100, 255])
        # # # ----------------------------------- Comment out for now to try different values 

        # Create mask for white lanes
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # Combine masks using bitwise_and()
        mask = cv2.bitwise_or(green_mask, white_mask)

        # Close any gaps in the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("raw", mask)
        cv2.waitKey(1)

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

        # Draw the detected lines on the original image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("isolated", isolated)
        cv2.waitKey(1)

        cv2.imshow("lines", cv_image)
        cv2.waitKey(1)

        row_indexes, col_indexes = np.nonzero(isolated)
        max_col = 0
        min_col = 0
        filtered_col_indexes = col_indexes[(col_indexes >= 800) & (col_indexes <= 1100)]

        # print(col_indexes)

        # Find find left side of edge and right side of edge
        if len(col_indexes) > 0:
            max_col = max(col_indexes)
            min_col = max(filtered_col_indexes)

        # print(min_col)
        print(max_col)

        # twist = Twist() 
        # twist.linear.x = 0.05
        # twist.angular.z = np.clip(self.pid.compute(max_col), -1, 1)
        # self.cmd_pub.publish(twist)

    


        # # --------------------------------sUPER GOOD WHITE LANE MASKING
        # # Define color thresholds for white lanes in HSV
        # lower_white = np.array([0, 0, 200])
        # upper_white = np.array([255, 30, 255])

        # # Create mask for white lanes
        # mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # # Close any gaps in the mask
        # kernel = np.ones((5,5), np.uouint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # # Apply mask to original image
        # masked_img = cv2.bitwise_and(cv_image, cv_image, mask=mask)
        # # -------------------------------UNCOMMENT THIS BLOCK 

        # # Define lower and upper bounds for green color in HSV
        # lower_green = np.array([25, 50, 50])
        # upper_green = np.array([85, 255, 255])

        # # Create a mask of all pixels within the green color range
        # mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # cv2.imshow("mask", masked_img)
        # cv2.waitKey(1)

        # # Apply the mask to the original image
        # filtered_image = cv2.bitwise_and(cv_image, cv_image, mask=mask)


    
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

    def region(self,image):
        height, width = image.shape

        polygon = np.array([[(int(width), height), (int(width),  
        int(height*0.92)), (int(width*0.05), int(height*0.92)), (int(0), height),]], np.int32)    

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
    def compute(self, x_coordinate):
        self.error = self.setpoint - (x_coordinate - 550)
        self.derivative_error = (self.error- self.error_last) / TIME_STEP
        self.error_last = self.error
        self.output = self.kp*self.error + self.kd*self.derivative_error
        self.output_scaled = np.divide(self.output,PID_SCALE)
        self.output_scaled = np.clip(self.output_scaled, -1, 1)
        return self.output_scaled
    def computeLeft(self, x_coordinate):
        self.error = self.setpoint - (x_coordinate + 575)
        self.derivative_error = (self.error- self.error_last) / (3*TIME_STEP)
        self.error_last = self.error
        self.output = self.kp*self.error + self.kd*self.derivative_error
        self.output_scaled = np.divide(self.output,PID_SCALE)
        self.output_scaled = np.clip(self.output_scaled, -1, 1)
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

