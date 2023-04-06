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
        self.speed_up = False
        self.start_speedup_timer = 0

        self.moving_car_detected = False
        self.car_frame_counter = 0
        self.prev_car_frame = None
    

        self.clock_sub = rospy.Subscriber('/clock',Clock, self.clock_callback)
        self.license_pub = rospy.Publisher('/license_plate',String,queue_size=10)
        time.sleep(0.2)

        
        self.license_pub.publish(String('Vlandy,pass,0,VLAND7')) #Start Time

        self.off_terrain = False

        self.min_col_detected = False
        self.count_rising_edge = 0





    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            height, width = hsv_image.shape[:2]
            crop_image = cv_image[int(height*0.6):height, 0:width]
            hsv_crop_image = hsv_image[int(height*0.6):height, 0:width]

            hsv_crop_image = cv2.bilateralFilter(hsv_crop_image,10,100,100)
            #define the lower and upper hsv values for the hsv colors
            lower_hsv = np.uint8(np.array([100, 0, 80]))
            upper_hsv = np.uint8(np.array([160, 70, 190]))

            # mask and extract the license plate
            mask = cv2.inRange(hsv_crop_image, lower_hsv, upper_hsv)

            mask_bin = mask.astype(np.uint8) * 255

            # Count the number of blue pixels in the ROI
            pixel_count = cv2.countNonZero(mask_bin)
            if(pixel_count>3300):
                _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                x, y, w, h, _ = stats[largest_label]
                result = crop_image[y:y+h, x:x+w]
                self.image_filename = f"images/{self.countResult}.jpg"
                self.countResult += 1
                # cv2.imwrite(self.image_filename, cv_image)
                # cv2.imshow("mask", result)
                # cv2.waitKey(500)

        except CvBridgeError as e:
            print(e)

        copy = np.copy(cv_image)

        # ----------------------------NON-HSV
        gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)

        blured_image = cv2.GaussianBlur(gray, (3,3), 0)

        edges = cv2.Canny(blured_image, 0, 50)

        isolated = self.region(edges)

        # cv2.imshow("edges",isolated)
        # cv2.waitKey(1)

        row_indexes, col_indexes = np.nonzero(isolated)
        max_col = 0
        min_col = 0

        # Find find left side of edge and right side of edge
        if len(col_indexes) > 0:
            max_col = max(col_indexes)
            min_col = min(col_indexes)

        print(min_col)

        twist = Twist()

        if self.start_timer < 0.5:
            twist.linear.x = -0.05
            twist.angular.z = 0 
            self.cmd_pub.publish(twist)
            print("case 1")
        elif self.start_timer >= 0.5 and self.start_timer < 3:
            twist.linear.x = 0
            twist.angular.z = 1
            self.cmd_pub.publish(twist)
            print("case 2")

        elif self.min_col_detected == False:
            twist.linear.x = 0
            twist.angular.z = 1
            self.cmd_pub.publish(twist)
            if min_col in range(30,75):
                self.min_col_detected = True
                twist.linear.x = 0
                twist.angular.z = 0
                self.cmd_pub.publish(twist)
            print("case 3")
        elif self.moving_car_detected == False:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                img_car = cv_image[450:720, 300: 700]
                # cv2.imshow("image", img_car)
                # cv2.waitKey(1)
                # # plt.show()
            except CvBridgeError as e:
                print(e)

            if self.car_frame_counter % 4 == 0:
                if self.prev_car_frame is not None:
                    # subtract the current frame from the previous frame
                    diff = cv2.absdiff(img_car, self.prev_car_frame)
                    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                    print("nonzero\n")
                    print(cv2.countNonZero(thresh))
                    
                    # check for movement by counting the number of non-zero pixels
                    if cv2.countNonZero(thresh) > 1000:
                        self.moving_car_detected = True
                        self.start_timer = 3
                        time.sleep(0.5)
                        print("Movement detected.")
                    else:
                        print("No movement detected.")
                        
                # update the previous frame
                self.prev_car_frame = img_car.copy()
            
            # increment the counter
            self.car_frame_counter += 1

            print("case 4")
        else:
            twist = Twist()
            twist.linear.x = 0.20
            if(min_col > 1000):
                min_col = 50
                self.count_rising_edge += 1
            twist.angular.z = self.pidLeft.computeLeft(min_col) * 1.2
            self.cmd_pub.publish(twist)

            print(self.count_rising_edge)
            print("case 5")

        self.start_timer += TIME_STEP
        print(self.start_timer)

        


    def clock_callback(self,msg):
        current_time = msg.clock

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
