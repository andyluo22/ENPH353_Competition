#! /usr/bin/env python3

import rospy
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import sys

CAM_IMAGE_HEIGHT = 720
CAM_IMAGE_WIDTH = 1280
TIME_STEP = 0.040
PID_SCALE = 250
LANE_WIDTH = 80 # about 1/13 of 1280 so i just used 80 which seems like a good give or take
RIGHT_MOST_X_COORDINATE = 1220
LEFT_MOST_X_COORDINATE = 60


class Controller:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/R1/pi_camera/image_raw', Image, self.image_callback)
        self.cmd_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        self.pid = PID(2.6, 0.35,RIGHT_MOST_X_COORDINATE) # KP = 2, KD = 0.5, and x speed = 0.25 below was rlly good
        self.pidLeft = PID(14,2,CAM_IMAGE_WIDTH/2)
        self.start_timer = 0
        self.count_license_plates = 0


    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # plt.show()
        except CvBridgeError as e:
            print(e)

        state = np.zeros(10, dtype=np.int)

        copy = np.copy(cv_image)

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

        print(max_col)
        print(min_col)

        cv2.imshow("raw", cv_image)
        cv2.waitKey(1)

        if self.start_timer < 1.8:
            twist = Twist()
            twist.linear.x = 0.17
            twist.angular.z = 0.6
            self.cmd_pub.publish(twist)

            self.start_timer += TIME_STEP
            print(self.start_timer)

        elif self.start_timer >= 1.8 and self.start_timer < 4:
            twist = Twist()
            twist.linear.x = 0.14
            twist.angular.z = np.clip(self.pidLeft.computeLeft(min_col), -1, 1)
            self.cmd_pub.publish(twist)

            self.start_timer += TIME_STEP
            print(self.start_timer)
        else: 
            twist = Twist() 
            twist.linear.x = 0.30
            twist.angular.z = self.pid.computeRight(max_col)
            self.cmd_pub.publish(twist)

            self.start_timer += TIME_STEP
            print(self.start_timer)


    def region(self,image):
        height, width = image.shape

        polygon = np.array([[(int(width), height), (int(width),  
        int(height*0.95)), (int(width*0.00), int(height*0.95)), (int(0), height),]], np.int32)    

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
        self.error = self.setpoint - (x_coordinate + 575)
        self.derivative_error = (self.error- self.error_last) / (3*TIME_STEP)
        self.error_last = self.error
        self.output = self.kp*self.error + self.kd*self.derivative_error
        self.output_scaled = np.divide(self.output,PID_SCALE)
        self.output_scaled = np.clip(self.output_scaled, -1, 1)
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