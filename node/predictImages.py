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

# model_path = '/home/fizzer/ros_ws/src/controller_pkg/node/modelGoodie.h5'
# model = tf.keras.models.load_model(model_path)

model_path = '/home/fizzer/ros_ws/src/controller_pkg/node/modelAction12.h5'
model = tf.keras.models.load_model(model_path)

model_license_path = '/home/fizzer/ros_ws/src/controller_pkg/node/VladasHotLittleModel.h5'
model_license = tf.keras.models.load_model(model_license_path)


class Controller:
    # def __init__(self):
    #     self.bridge = CvBridge()
    #     self.image_sub = rospy.Subscriber(
    #         '/R1/pi_camera/image_raw', Image, self.image_callback)
    #     self.cmd_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
    #     self.pid = PID(14,2,CAM_IMAGE_WIDTH/2)

    #     # Create a timer that will call the update_angular_vel function every 100 ms
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/R1/pi_camera/image_raw', Image, self.image_callback)
        self.cmd_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        self.pid = PID(2.6, 0.20,RIGHT_MOST_X_COORDINATE) # KP = 2, KD = 0.5, and x speed = 0.25 below was rlly good
        self.pidLeft = PID(2.4, 0.10, LEFT_MOST_X_COORDINATE)
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

        self.off_terrain = False
        self.min_col_detected = False
        self.count_rising_edge = 0
        self.state_inner_done = False
        self.countResult = 0
        self.state_detect_pedestrian = False
        self.count_red_lines = 0
        self.pedestrian_detected_timer = 0

        self.grass_terrain_detected = False


        self.moving_car_detected = False
        self.car_frame_counter = 0
        self.prev_car_frame = None
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.license_pub = rospy.Publisher('/license_plate',String,queue_size=10)
        time.sleep(0.2)
        
        self.license_pub.publish(String('Vlandy,pass,0,VLAND7')) #Start Time



    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            height, width = hsv_image.shape[:2]
            crop_image = cv_image[int(height*0.6):height, 0:width]
            crop_image = cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
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

                # cv2.imshow("Result", result)
                # cv2.waitKey(1)
                # self.image_filename = f"images/{self.countResult}.jpg"
                char1 = result[0:h, 0:int(w*0.8/3)]
                print(char1.shape)

                # cv2.imshow("char1", char1)
                # cv2.waitKey(1)

                char1 = cv2.resize(char1, (100, 120))
                char1 = np.expand_dims(char1, axis=0)
                # print(char1.shape)
                # char2 = result[0:h, int(w*0.9/3): int(w*0.9/2)]
                # char3 = result[0:h, int(w*1.2/2): int(w*2.3/3)]
                # # char4 = result[0:h, int(w*2.3/3): w]

                char1_pred = model_license.predict(char1)[0]

                print(char1_pred)
                char1String = self.onehot_to_string(char1_pred)

                index = np.argmax(char1String)

                print(index)
                # char2_pred = model_license.predict(char2)
                # char2String = self.onehot_to_string(char2_pred)
                # char3_pred = model_license.predict(char3)
                # char3String = self.onehot_to_string(char3_pred)
                # char4_pred = model_license.predict(char4)
                # char4String = self.onehot_to_string(char4_pred)

                print(char1String)



                
                self.countResult += 1
                
                # cv2.imwrite(self.image_filename, cv_image)
                # cv2.imshow("mask", result)
                # cv2.waitKey(1)
            cv_image = self.preProcessing(cv_image)
        
        except CvBridgeError as e:
            print(e)

        # try:
        #     # Pass the image through the model and get the predicted class
        #     prediction = model.predict(np.array([cv_image]))

        #     max_index = np.argmax(prediction)
        #     if max_index == 0:
        #         # Left
        #         angular_vel = 0.64
        #     elif max_index == 1:
        #         # Straight
        #         angular_vel = 0.0
        #     else:
        #         # Right
        #         angular_vel = -0.64

        #     linear_vel = 0.11
        #     print(prediction)
            
        #     # # Map the predicted class to the corresponding steering angle
        #     # steering_angles = {'L': 0.86, 'S': 0.0, 'R': -0.86}
        #     # angular_vel = steering_angles[pred_class]
        #     # linear_vel = 0.15
        # except Exception as e:
        #     print("Error predicting: ", str(e))

        # twist = Twist()
        # twist.linear.x = linear_vel
        # twist.angular.z = angular_vel
        # self.cmd_pub.publish(twist)


    def region(self,image):
        height, width = image.shape

        polygon = np.array([[(int(width), height), (int(width),  
        int(height*0.92)), (int(width*0.05), int(height*0.92)), (int(0), height),]], np.int32)    

        mask = np.zeros_like(image)
    
        mask = cv2.fillPoly(mask, polygon, 255)
        mask = cv2.bitwise_and(image, mask)

        return mask
    
    def preProcessing(self,img):
        img = img[400:720, 640:1280]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img,(3,3),0)
        img = cv2.resize(img, (200,66))
        img = img / 255
        return img
    
    
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