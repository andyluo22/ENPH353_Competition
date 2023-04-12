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
LOWER_HSV = np.uint8(np.array([100, 0, 80]))
UPPER_HSV = np.uint8(np.array([160, 70, 190]))

CAR_ORDER = [7,8,3,4,5,6,1,2]

model_path = '/home/fizzer/ros_ws/src/controller_pkg/node/modelAction12.h5'
model = tf.keras.models.load_model(model_path)

# OG Model
# model_license_path = '/home/fizzer/ros_ws/src/controller_pkg/node/VladasHotLittleModel.h5'
# model_license = tf.keras.models.load_model(model_license_path)

# New Font First GOOOd 
# model_license_path = '/home/fizzer/ros_ws/src/controller_pkg/node/VladasHotLittleModelNewFontWhoDisUptrain.h5'
# model_license = tf.keras.models.load_model(model_license_path)

#
model_license_path = '/home/fizzer/ros_ws/src/controller_pkg/node/VladasHotLittleModelNewFontWhoDisUptrainTimesTwo.h5'
model_license = tf.keras.models.load_model(model_license_path)

# # New Font Third 
# model_license_path = '/home/fizzer/ros_ws/src/controller_pkg/node/SheIsTiredOfTraining.h5'
# model_license = tf.keras.models.load_model(model_license_path)


class Controller:
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
        self.time_of_last_license_plate = 0

        self.off_terrain = False
        self.min_col_detected = False
        self.count_rising_edge = 0
        self.state_inner_done = False
        self.countResult = 0
        self.state_detect_pedestrian = False
        self.count_red_lines = 0
        self.pedestrian_detected_timer = 0

        # License Plate Counts 

        self.grass_terrain_detected = False

        # Store times when we stop to detect cars
        self.countGrassCarOne = 0
        self.countGrassCarTwo = 0
        self.grassCarOneTimer = 0
        self.grassCarTwoTimer = 0
        self.countGrassCarThree = 0

        self.moving_car_detected = False
        self.car_frame_counter = 0
        self.prev_car_frame = None
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.clock_sub = rospy.Subscriber('/clock',Clock, self.clock_callback)
        self.license_pub = rospy.Publisher('/license_plate',String,queue_size=10)
        time.sleep(0.2)
        
        self.license_pub.publish(String('Vlandy,Shrek,0,VLAND7')) #Start Time






    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            if self.grass_terrain_detected == False:
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
                if(pixel_count>2800 and time.time() - self.time_of_last_license_plate > 3):
                    self.time_of_last_license_plate = time.time()
                    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    x, y, w, h, _ = stats[largest_label]
                    result = crop_image[y:y+h, x:x+w]
                    if self.count_license_plates==1:
                        current_mean = np.mean(result)
                        alpha = 1.0
                        beta = 140 - current_mean
                        result = cv2.addWeighted(result, alpha, np.zeros(result.shape, result.dtype), 0, beta)
                        # Clip the resulting image to ensure valid pixel values
                        result =  np.clip(result, 0, 255).astype(np.uint8)
                    elif self.count_license_plates ==0:
                        current_mean = np.mean(result)
                        alpha = 1.8
                        beta = 90 - current_mean
                        result = cv2.addWeighted(result, alpha, np.zeros(result.shape, result.dtype), 0, beta)
                        # Clip the resulting image to ensure valid pixel values
                        result =  np.clip(result, 0, 255).astype(np.uint8)
                    else:
                        current_mean = np.mean(result)
                        alpha = 1.7
                        beta = 100 - current_mean
                        result = cv2.addWeighted(result, alpha, np.zeros(result.shape, result.dtype), 0, beta)
                        # Clip the resulting image to ensure valid pixel values
                        result =  np.clip(result, 0, 255).astype(np.uint8)


                    cv2.imshow("license plate", result)
                    cv2.waitKey(1)
                    self.count_license_plates += 1

                    _, mask1 = cv2.threshold(result, 85, 255, cv2.THRESH_BINARY)

                    mask1 = cv2.bitwise_not(mask1)

                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1)
                    sizes = stats[1:, -1]
                    component_indices = np.argsort(sizes)[::-1][:4] # select top four largest components

                                        
                    chars = []
                    x_vals = []
                    for i in component_indices:
                        x, y, w, h = stats[i+1, :4]
                        char = result[y-5:y+h+5, x-5:x+w+5]
                        x_vals.append(x)
                        chars.append(char)

                    sorted_x = np.argsort(x_vals)
                    sorted_chars = [chars[i] for i in sorted_x]

                    char1 = cv2.resize(sorted_chars[0], (30,36))
                    char2 = cv2.resize(sorted_chars[1], (30,36))
                    char3 = cv2.resize(sorted_chars[2], (30,36))
                    char4 = cv2.resize(sorted_chars[3], (30,36))

                    # cv2.imshow("char1",char1)
                    # cv2.waitKey(1)
                    # cv2.imshow("char2",char2)
                    # cv2.waitKey(1)
                    # cv2.imshow("char3",char3)
                    # cv2.waitKey(1)
                    # cv2.imshow("char4",char4)
                    # cv2.waitKey(1)


                    char1 = np.expand_dims(char1, axis=0)
                    char2 = np.expand_dims(char2, axis=0)
                    char3 = np.expand_dims(char3, axis=0)
                    char4 = np.expand_dims(char4, axis=0)

                    char1_pred = model_license.predict(char1)
                    char2_pred = model_license.predict(char2)
                    char3_pred = model_license.predict(char3)
                    char4_pred = model_license.predict(char4)

                    index1 = np.argmax(char1_pred)
                    index2 = np.argmax(char2_pred)
                    index3 = np.argmax(char3_pred)
                    index4 = np.argmax(char4_pred)

                    string1 = self.onehot_to_string(index1)
                    string2 = self.onehot_to_string(index2)
                    string3 = self.onehot_to_string(index3)
                    string4 = self.onehot_to_string(index4)
                    
                    plate = string1 + string2 + string3 + string4

                    message = "TeamRed,multi21,{},{}".format(CAR_ORDER[self.count_license_plates-1], plate)
                    self.license_pub.publish(message)
                    


                    print("PREDICTED ------------------\n")
                    print(string1 + string2 + string3 + string4)
                    # print("PREDICTED ------------------\n")
                    # print(index1)
                    # print(index2)
                    # print(index3)
                    # print(index4)

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

        if self.grass_terrain_detected == True:
            try:
                # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                cv_pred = self.preProcessing(cv_image)
            
            except CvBridgeError as e:
                print(e)

            if self.start_timer < 18.5:
                try:
                    prediction = model.predict(np.array([cv_pred]))

                    max_index = np.argmax(prediction)
                    if max_index == 0:
                        # Left
                        angular_vel = 0.7
                    elif max_index == 1:
                        # Straight
                        angular_vel = 0.0
                    else:
                        # Right
                        angular_vel = -0.6

                    linear_vel = 0.10
                    print(prediction)
                    twist = Twist()
                    twist.linear.x = linear_vel
                    twist.angular.z = angular_vel
                    self.cmd_pub.publish(twist) 

                    if int(self.start_timer)==3 and self.countGrassCarOne == 0:
                        # self.countGrassCarOne += 1
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
                        if(pixel_count>2500 and time.time() - self.time_of_last_license_plate > 3):
                            self.time_of_last_license_plate = time.time()
                            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
                            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                            x, y, w, h, _ = stats[largest_label]
                            result = crop_image[y:y+h, x:x+w]

                            current_mean = np.mean(result)
                            alpha = 1.0
                            beta = 120.0 - current_mean


                            result = cv2.addWeighted(result, alpha, np.zeros(result.shape, result.dtype), 0, beta)

                            # Clip the resulting image to ensure valid pixel values
                            result =  np.clip(result, 0, 255).astype(np.uint8)

                            cv2.imshow("license plate", result)
                            cv2.waitKey(1)
                            self.count_license_plates += 1

                            _, mask1 = cv2.threshold(result, 80, 255, cv2.THRESH_BINARY)

                            mask1 = cv2.bitwise_not(mask1)

                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1)
                            sizes = stats[1:, -1]
                            component_indices = np.argsort(sizes)[::-1][:4] # select top four largest components

                                                
                            chars = []
                            x_vals = []
                            for i in component_indices:
                                x, y, w, h = stats[i+1, :4]
                                char = result[y-5:y+h+5, x-5:x+w+5]
                                x_vals.append(x)
                                chars.append(char)

                            sorted_x = np.argsort(x_vals)
                            sorted_chars = [chars[i] for i in sorted_x]

                            char1 = cv2.resize(sorted_chars[0], (30,36))
                            char2 = cv2.resize(sorted_chars[1], (30,36))
                            char3 = cv2.resize(sorted_chars[2], (30,36))
                            char4 = cv2.resize(sorted_chars[3], (30,36))

                            # cv2.imshow("char1",char1)
                            # cv2.waitKey(1)
                            # cv2.imshow("char2",char2)
                            # cv2.waitKey(1)
                            # cv2.imshow("char3",char3)
                            # cv2.waitKey(1)
                            # cv2.imshow("char4",char4)
                            # cv2.waitKey(1)


                            char1 = np.expand_dims(char1, axis=0)
                            char2 = np.expand_dims(char2, axis=0)
                            char3 = np.expand_dims(char3, axis=0)
                            char4 = np.expand_dims(char4, axis=0)

                            char1_pred = model_license.predict(char1)
                            char2_pred = model_license.predict(char2)
                            char3_pred = model_license.predict(char3)
                            char4_pred = model_license.predict(char4)

                            index1 = np.argmax(char1_pred)
                            index2 = np.argmax(char2_pred)
                            index3 = np.argmax(char3_pred)
                            index4 = np.argmax(char4_pred)

                            string1 = self.onehot_to_string(index1)
                            string2 = self.onehot_to_string(index2)
                            string3 = self.onehot_to_string(index3)
                            string4 = self.onehot_to_string(index4)

                            plate = string1 + string2 + string3 + string4

                            message = "TeamRed,multi21,{},{}".format(CAR_ORDER[self.count_license_plates-1], plate)
                            self.license_pub.publish(message)

                            print("PREDICTED ------------------\n")
                            print(string1 + string2 + string3 + string4)

                    elif int(self.start_timer)==11 and self.countGrassCarTwo == 0:
                        # self.countGrassCarTwo += 1
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
                        if(pixel_count>2900 and time.time() - self.time_of_last_license_plate > 3):
                            self.time_of_last_license_plate = time.time()
                            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
                            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                            x, y, w, h, _ = stats[largest_label]
                            result = crop_image[y:y+h, x:x+w]

                            current_mean = np.mean(result)
                            alpha = 1.0
                            beta = 120.0 - current_mean


                            result = cv2.addWeighted(result, alpha, np.zeros(result.shape, result.dtype), 0, beta)

                            # Clip the resulting image to ensure valid pixel values
                            result =  np.clip(result, 0, 255).astype(np.uint8)

                            cv2.imshow("license plate", result)
                            cv2.waitKey(1)

                            self.count_license_plates += 1

                            _, mask1 = cv2.threshold(result, 80, 255, cv2.THRESH_BINARY)

                            mask1 = cv2.bitwise_not(mask1)

                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1)
                            sizes = stats[1:, -1]
                            component_indices = np.argsort(sizes)[::-1][:4] # select top four largest components

                                                
                            chars = []
                            x_vals = []
                            for i in component_indices:
                                x, y, w, h = stats[i+1, :4]
                                char = result[y-5:y+h+5, x-5:x+w+5]
                                x_vals.append(x)
                                chars.append(char)

                            sorted_x = np.argsort(x_vals)
                            sorted_chars = [chars[i] for i in sorted_x]

                            char1 = cv2.resize(sorted_chars[0], (30,36))
                            char2 = cv2.resize(sorted_chars[1], (30,36))
                            char3 = cv2.resize(sorted_chars[2], (30,36))
                            char4 = cv2.resize(sorted_chars[3], (30,36))

                            # cv2.imshow("char1",char1)
                            # cv2.waitKey(1)
                            # cv2.imshow("char2",char2)
                            # cv2.waitKey(1)
                            # cv2.imshow("char3",char3)
                            # cv2.waitKey(1)
                            # cv2.imshow("char4",char4)
                            # cv2.waitKey(1)

                            char1 = np.expand_dims(char1, axis=0)
                            char2 = np.expand_dims(char2, axis=0)
                            char3 = np.expand_dims(char3, axis=0)
                            char4 = np.expand_dims(char4, axis=0)

                            char1_pred = model_license.predict(char1)
                            char2_pred = model_license.predict(char2)
                            char3_pred = model_license.predict(char3)
                            char4_pred = model_license.predict(char4)

                            index1 = np.argmax(char1_pred)
                            index2 = np.argmax(char2_pred)
                            index3 = np.argmax(char3_pred)
                            index4 = np.argmax(char4_pred)

                            string1 = self.onehot_to_string(index1)
                            string2 = self.onehot_to_string(index2)
                            string3 = self.onehot_to_string(index3)
                            string4 = self.onehot_to_string(index4)

                            plate = string1 + string2 + string3 + string4

                            message = "TeamRed,multi21,{},{}".format(CAR_ORDER[self.count_license_plates-1], plate)
                            self.license_pub.publish(message)

                            print("PREDICTED ------------------\n")
                            print(string1 + string2 + string3 + string4)
                            # print("PREDICTED ------------------\n")
                            # print(index1)
                            # print(index2)
                            # print(index3)
                            # print(index4)
                            self.countGrassCarTwo += 1

                    elif int(self.start_timer)==16 and self.countGrassCarThree == 0:
                        # self.countGrassCarTwo += 1
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
                        if(pixel_count>2700 and time.time() - self.time_of_last_license_plate > 3):
                            self.time_of_last_license_plate = time.time()
                            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
                            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                            x, y, w, h, _ = stats[largest_label]
                            result = crop_image[y:y+h, x:x+w]

                            current_mean = np.mean(result)
                            alpha = 1.0
                            beta = 110.0 - current_mean


                            result = cv2.addWeighted(result, alpha, np.zeros(result.shape, result.dtype), 0, beta)

                            # Clip the resulting image to ensure valid pixel values
                            result =  np.clip(result, 0, 255).astype(np.uint8)

                            cv2.imshow("license plate", result)
                            cv2.waitKey(1)

                            self.count_license_plates += 1

                            _, mask1 = cv2.threshold(result, 80, 255, cv2.THRESH_BINARY)

                            mask1 = cv2.bitwise_not(mask1)

                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1)
                            sizes = stats[1:, -1]
                            component_indices = np.argsort(sizes)[::-1][:4] # select top four largest components

                                                
                            chars = []
                            x_vals = []
                            for i in component_indices:
                                x, y, w, h = stats[i+1, :4]
                                char = result[y-5:y+h+5, x-5:x+w+5]
                                x_vals.append(x)
                                chars.append(char)

                            sorted_x = np.argsort(x_vals)
                            sorted_chars = [chars[i] for i in sorted_x]

                            char1 = cv2.resize(sorted_chars[0], (30,36))
                            char2 = cv2.resize(sorted_chars[1], (30,36))
                            char3 = cv2.resize(sorted_chars[2], (30,36))
                            char4 = cv2.resize(sorted_chars[3], (30,36))

                            # cv2.imshow("char1",char1)
                            # cv2.waitKey(1)
                            # cv2.imshow("char2",char2)
                            # cv2.waitKey(1)
                            # cv2.imshow("char3",char3)
                            # cv2.waitKey(1)
                            # cv2.imshow("char4",char4)
                            # cv2.waitKey(1)

                            char1 = np.expand_dims(char1, axis=0)
                            char2 = np.expand_dims(char2, axis=0)
                            char3 = np.expand_dims(char3, axis=0)
                            char4 = np.expand_dims(char4, axis=0)

                            char1_pred = model_license.predict(char1)
                            char2_pred = model_license.predict(char2)
                            char3_pred = model_license.predict(char3)
                            char4_pred = model_license.predict(char4)

                            index1 = np.argmax(char1_pred)
                            index2 = np.argmax(char2_pred)
                            index3 = np.argmax(char3_pred)
                            index4 = np.argmax(char4_pred)

                            string1 = self.onehot_to_string(index1)
                            string2 = self.onehot_to_string(index2)
                            string3 = self.onehot_to_string(index3)
                            string4 = self.onehot_to_string(index4)

                            plate = string1 + string2 + string3 + string4

                            message = "TeamRed,multi21,{},{}".format(CAR_ORDER[self.count_license_plates-1], plate)
                            self.license_pub.publish(message)

                            print("PREDICTED ------------------\n")
                            print(string1 + string2 + string3 + string4)
                            # print("PREDICTED ------------------\n")
                            # print(index1)
                            # print(index2)
                            # print(index3)
                            # print(index4)
                            self.countGrassCarThree+= 1
                    
                # # Map the predicted class to the corresponding steering angle
                # steering_angles = {'L': 0.86, 'S': 0.0, 'R': -0.86}
                # angular_vel = steering_angles[pred_class]
                # linear_vel = 0.15
                except Exception as e:
                    print("Error predicting: ", str(e))

                # twist = Twist()
                # twist.linear.x = linear_vel
                # twist.angular.z = angular_vel
                # self.cmd_pub.publish(twist)
            # Pass the image through the model and get the predicted class
            elif (self.start_timer >= 19 and self.start_timer < 21.5) or self.start_timer >= 23:
                twist = Twist() 
                twist.linear.x = 0.22
                twist.angular.z = self.pid.computeRight(max_col)
                self.cmd_pub.publish(twist)
                print("about to finish outerloop")
                #Detect Red Line Process
                img_red = cv_image[700:720, 900: 1100]
                # Convert the image to HSV color space
                hsv_img_red = cv2.cvtColor(img_red, cv2.COLOR_BGR2HSV)

                # Define the range of red color in HSV color space
                lower_red = np.array([0, 50, 50])
                upper_red = np.array([10, 255, 255])

                # Create a mask using the range of red color
                mask1_red = cv2.inRange(hsv_img_red, lower_red, upper_red)

                # Define the range of red color in HSV color space
                lower_red = np.array([170, 50, 50])
                upper_red = np.array([180, 255, 255])

                # Create another mask using the range of red color
                mask2_red = cv2.inRange(hsv_img_red, lower_red, upper_red)

                # Combine the masks
                mask_red = mask1_red + mask2_red

                # Apply the mask to the original image
                red_img = cv2.bitwise_and(img_red, img_red, mask=mask_red)

                gray_red = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)

                ##remove noise using gaussian blur 
                blured_image_red = cv2.GaussianBlur(gray_red, (3,3), 0)

                ##detect edges from luminosity change 
                edges_red = cv2.Canny(blured_image_red, 100, 200)

                red_line_points = np.count_nonzero(edges_red)

                # self.countGrassCarOne += 1
                height, width = hsv_image.shape[:2]
                crop_image = cv_image[int(height*0.3):height, 0:width]
                crop_image = cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
                hsv_crop_image = hsv_image[int(height*0.3):height, 0:width]

                hsv_crop_image = cv2.bilateralFilter(hsv_crop_image,10,100,100)
                #define the lower and upper hsv values for the hsv colors
                lower_hsv = np.uint8(np.array([100, 0, 80]))
                upper_hsv = np.uint8(np.array([160, 70, 190]))

                # mask and extract the license plate
                mask = cv2.inRange(hsv_crop_image, lower_hsv, upper_hsv)

                mask_bin = mask.astype(np.uint8) * 255

                # Count the number of blue pixels in the ROI
                pixel_count = cv2.countNonZero(mask_bin)
                if self.count_license_plates == 5:
                    pixel_min =2800
                else:
                    pixel_min =2800
                if(pixel_count> pixel_min and time.time() - self.time_of_last_license_plate > 0.75):
                    self.time_of_last_license_plate = time.time()
                    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    x, y, w, h, _ = stats[largest_label]
                    result = crop_image[y:y+h, x:x+w]

                    if self.count_license_plates == 5:
                        current_mean = np.mean(result)
                        alpha = 1.7
                        beta = 80.0 - current_mean


                        result = cv2.addWeighted(result, alpha, np.zeros(result.shape, result.dtype), 0, beta)

                        # Clip the resulting image to ensure valid pixel values
                        result =  np.clip(result, 0, 255).astype(np.uint8)
                    else:
                        current_mean = np.mean(result)
                        alpha = 1.7
                        beta = 80 - current_mean
                        result = cv2.addWeighted(result, alpha, np.zeros(result.shape, result.dtype), 0, beta)
                        # Clip the resulting image to ensure valid pixel values
                        result =  np.clip(result, 0, 255).astype(np.uint8)


                    cv2.imshow("license plate", result)
                    cv2.waitKey(1)
                    self.count_license_plates += 1

                    _, mask1 = cv2.threshold(result, 80, 255, cv2.THRESH_BINARY)

                    mask1 = cv2.bitwise_not(mask1)

                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1)
                    sizes = stats[1:, -1]
                    component_indices = np.argsort(sizes)[::-1][:4] # select top four largest components

                                        
                    chars = []
                    x_vals = []
                    for i in component_indices:
                        x, y, w, h = stats[i+1, :4]
                        char = result[y-5:y+h+5, x-5:x+w+5]
                        x_vals.append(x)
                        chars.append(char)

                    sorted_x = np.argsort(x_vals)
                    sorted_chars = [chars[i] for i in sorted_x]

                    char1 = cv2.resize(sorted_chars[0], (30,36))
                    char2 = cv2.resize(sorted_chars[1], (30,36))
                    char3 = cv2.resize(sorted_chars[2], (30,36))
                    char4 = cv2.resize(sorted_chars[3], (30,36))

                    # cv2.imshow("char1",char1)
                    # cv2.waitKey(1)
                    # cv2.imshow("char2",char2)
                    # cv2.waitKey(1)
                    # cv2.imshow("char3",char3)
                    # cv2.waitKey(1)
                    # cv2.imshow("char4",char4)
                    # cv2.waitKey(1)

                    char1 = np.expand_dims(char1, axis=0)
                    char2 = np.expand_dims(char2, axis=0)
                    char3 = np.expand_dims(char3, axis=0)
                    char4 = np.expand_dims(char4, axis=0)

                    char1_pred = model_license.predict(char1)
                    char2_pred = model_license.predict(char2)
                    char3_pred = model_license.predict(char3)
                    char4_pred = model_license.predict(char4)

                    index1 = np.argmax(char1_pred)
                    index2 = np.argmax(char2_pred)
                    index3 = np.argmax(char3_pred)
                    index4 = np.argmax(char4_pred)

                    string1 = self.onehot_to_string(index1)
                    string2 = self.onehot_to_string(index2)
                    string3 = self.onehot_to_string(index3)
                    string4 = self.onehot_to_string(index4)

                    plate = string1 + string2 + string3 + string4

                    message = "TeamRed,multi21,{},{}".format(CAR_ORDER[self.count_license_plates-1], plate)
                    self.license_pub.publish(message)
                    

                    print("PREDICTED ------------------\n")
                    print(string1 + string2 + string3 + string4)
                    # print("PREDICTED ------------------\n")
                    # print(index1)
                    # print(index2)
                    # print(index3)
                    # print(index4)

                if red_line_points > 0:
                    twist = Twist()
                    twist.linear.x = 0
                    twist.angular.z = 0
                    self.cmd_pub.publish(twist)
                    self.license_pub.publish(String('Vlandy,Shrek,-1,VLANDFinished')) #End Time
                    print("The end")
                    time.sleep(20)
            else:
                twist = Twist()
                twist.linear.x = 0.21
                if(min_col > 1000):
                    min_col = 50
                    self.count_rising_edge += 1
                twist.angular.z = self.pidLeft.computeLeft(min_col) * 1.4
                self.cmd_pub.publish(twist)
                # self.countGrassCarOne += 1
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
                if(pixel_count>2900 and time.time() - self.time_of_last_license_plate > 3):
                    self.time_of_last_license_plate = time.time()
                    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    x, y, w, h, _ = stats[largest_label]
                    result = crop_image[y:y+h, x:x+w]

                    current_mean = np.mean(result)
                    alpha = 2.0
                    beta = 80 - current_mean


                    result = cv2.addWeighted(result, alpha, np.zeros(result.shape, result.dtype), 0, beta)

                    # Clip the resulting image to ensure valid pixel values
                    result =  np.clip(result, 0, 255).astype(np.uint8)

                    cv2.imshow("license plate", result)
                    cv2.waitKey(1)
                    self.count_license_plates += 1

                    _, mask1 = cv2.threshold(result, 80, 255, cv2.THRESH_BINARY)

                    mask1 = cv2.bitwise_not(mask1)

                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1)
                    sizes = stats[1:, -1]
                    component_indices = np.argsort(sizes)[::-1][:4] # select top four largest components

                                        
                    chars = []
                    x_vals = []
                    for i in component_indices:
                        x, y, w, h = stats[i+1, :4]
                        char = result[y-5:y+h+5, x-5:x+w+5]
                        x_vals.append(x)
                        chars.append(char)

                    sorted_x = np.argsort(x_vals)
                    sorted_chars = [chars[i] for i in sorted_x]

                    char1 = cv2.resize(sorted_chars[0], (30,36))
                    char2 = cv2.resize(sorted_chars[1], (30,36))
                    char3 = cv2.resize(sorted_chars[2], (30,36))
                    char4 = cv2.resize(sorted_chars[3], (30,36))

                    # cv2.imshow("char1",char1)
                    # cv2.waitKey(1)
                    # cv2.imshow("char2",char2)
                    # cv2.waitKey(1)
                    # cv2.imshow("char3",char3)
                    # cv2.waitKey(1)
                    # cv2.imshow("char4",char4)
                    # cv2.waitKey(1)

                    char1 = np.expand_dims(char1, axis=0)
                    char2 = np.expand_dims(char2, axis=0)
                    char3 = np.expand_dims(char3, axis=0)
                    char4 = np.expand_dims(char4, axis=0)

                    char1_pred = model_license.predict(char1)
                    char2_pred = model_license.predict(char2)
                    char3_pred = model_license.predict(char3)
                    char4_pred = model_license.predict(char4)

                    index1 = np.argmax(char1_pred)
                    index2 = np.argmax(char2_pred)
                    index3 = np.argmax(char3_pred)
                    index4 = np.argmax(char4_pred)

                    string1 = self.onehot_to_string(index1)
                    string2 = self.onehot_to_string(index2)
                    string3 = self.onehot_to_string(index3)
                    string4 = self.onehot_to_string(index4)

                    plate = string1 + string2 + string3 + string4

                    message = "TeamRed,multi21,{},{}".format(CAR_ORDER[self.count_license_plates-1], plate)
                    self.license_pub.publish(message)

                    print("PREDICTED ------------------\n")
                    print(string1 + string2 + string3 + string4)
                    # print("PREDICTED ------------------\n")
                    # print(index1)
                    # print(index2)
                    # print(index3)
                    # print(index4)

        elif self.state_detect_pedestrian == False and self.count_red_lines == 0:
            if self.start_timer < 0.5:
                twist.linear.x = -0.06
                twist.angular.z = 0 
                self.cmd_pub.publish(twist)
                print("case 1")
            elif self.start_timer >= 0.5 and self.start_timer < 3:
                twist.linear.x = 0
                twist.angular.z = 1
                self.cmd_pub.publish(twist)
                print("case 2")

            elif self.min_col_detected == False:
                twist.linear.x = 0.01
                twist.angular.z = 0.5
                self.cmd_pub.publish(twist)
                if min_col in range(30,75):
                    self.min_col_detected = True
                    twist.linear.x = 0
                    twist.angular.z = 0
                    self.cmd_pub.publish(twist)
                    time.sleep(1)
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
                        if cv2.countNonZero(thresh) > 4000:
                            self.moving_car_detected = True
                            self.start_timer = 3
                            time.sleep(0.8)
                            print("Movement detected.")
                        else:
                            print("No movement detected.")
                            
                    # update the previous frame
                    self.prev_car_frame = img_car.copy()
                
                # increment the counter
                self.car_frame_counter += 1

                print("case 4")
                
            elif self.start_timer >= 3 and self.start_timer < 9.2:
                twist = Twist()
                twist.linear.x = 0.21
                if(min_col > 1000):
                    min_col = 50
                    self.count_rising_edge += 1
                twist.angular.z = self.pidLeft.computeLeft(min_col) * 1.2
                self.cmd_pub.publish(twist)

                print(self.count_rising_edge)
                print("case 4")

            elif self.start_timer >= 9.2 and self.start_timer < 11:
                twist = Twist() 
                twist.linear.x = 0.24
                twist.angular.z = self.pid.computeRight(max_col)
                self.cmd_pub.publish(twist)
                print("case 5")
            
            elif self.start_timer >= 11 and self.state_inner_done == False:
                #Detect Red Line Process
                img_red = cv_image[700:720, 900: 1100]
                # Convert the image to HSV color space
                hsv_img_red = cv2.cvtColor(img_red, cv2.COLOR_BGR2HSV)

                # Define the range of red color in HSV color space
                lower_red = np.array([0, 50, 50])
                upper_red = np.array([10, 255, 255])

                # Create a mask using the range of red color
                mask1_red = cv2.inRange(hsv_img_red, lower_red, upper_red)

                # Define the range of red color in HSV color space
                lower_red = np.array([170, 50, 50])
                upper_red = np.array([180, 255, 255])

                # Create another mask using the range of red color
                mask2_red = cv2.inRange(hsv_img_red, lower_red, upper_red)

                # Combine the masks
                mask_red = mask1_red + mask2_red

                # Apply the mask to the original image
                red_img = cv2.bitwise_and(img_red, img_red, mask=mask_red)

                gray_red = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)

                ##remove noise using gaussian blur 
                blured_image_red = cv2.GaussianBlur(gray_red, (3,3), 0)

                ##detect edges from luminosity change 
                edges_red = cv2.Canny(blured_image_red, 100, 200)

                red_line_points = np.count_nonzero(edges_red)
                
                if red_line_points > 0:
                    self.state_inner_done = True

                twist = Twist()
                twist.linear.x = 0.085
                if(min_col > 1000):
                    min_col = 0
                twist.angular.z = self.pidLeft.computeLeft(min_col) * 0.95
                self.cmd_pub.publish(twist)

                print("Case 6")
            else:
                twist = Twist()
                twist.linear.x = 0
                twist.angular.z = 0
                self.cmd_pub.publish(twist)
                self.state_detect_pedestrian = True
                self.count_red_lines += 1
                # time.sleep(1)

        elif self.state_detect_pedestrian == True:
            twist = Twist()
            twist.linear.x = 0
            twist.angular.z = 0
            self.cmd_pub.publish(twist)
            img_ped = cv_image[250:500, 550:650]

            gray = cv2.cvtColor(img_ped, cv2.COLOR_RGB2GRAY)

            # detect people in the image
            # returns the bounding boxes for the detected objects
            boxes, weights = self.hog.detectMultiScale(img_ped, winStride=(8,8) )

            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

            if len(boxes) == 0:
                print("No pedestrians detected.")
            else:
                self.state_detect_pedestrian = False
                self.pedestrian_detected_timer = self.start_timer
                print("Go!")


        else:
            twist = Twist() 
            twist.linear.x = 0.20
            twist.angular.z = self.pid.computeRight(max_col)
            self.cmd_pub.publish(twist)

            if ((self.start_timer - self.pedestrian_detected_timer) > 5 and self.count_red_lines < 2):
                 #Detect Red Line Process
                img_red = cv_image[700:720, 900: 1100]
                # Convert the image to HSV color space
                hsv_img_red = cv2.cvtColor(img_red, cv2.COLOR_BGR2HSV)

                # Define the range of red color in HSV color space
                lower_red = np.array([0, 50, 50])
                upper_red = np.array([10, 255, 255])

                # Create a mask using the range of red color
                mask1_red = cv2.inRange(hsv_img_red, lower_red, upper_red)

                # Define the range of red color in HSV color space
                lower_red = np.array([170, 50, 50])
                upper_red = np.array([180, 255, 255])

                # Create another mask using the range of red color
                mask2_red = cv2.inRange(hsv_img_red, lower_red, upper_red)

                # Combine the masks
                mask_red = mask1_red + mask2_red

                # Apply the mask to the original image
                red_img = cv2.bitwise_and(img_red, img_red, mask=mask_red)

                gray_red = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)

                ##remove noise using gaussian blur 
                blured_image_red = cv2.GaussianBlur(gray_red, (3,3), 0)

                ##detect edges from luminosity change 
                edges_red = cv2.Canny(blured_image_red, 100, 200)

                red_line_points = np.count_nonzero(edges_red)
            
                if red_line_points > 0:
                    self.prev_frame = None
                    twist = Twist()
                    twist.linear.x = 0
                    twist.angular.z = 0
                    self.cmd_pub.publish(twist)
                    self.state_detect_pedestrian = True
                    self.count_red_lines += 1
                    # time.sleep(2)
                    print("Last Red Line Count")
                    print(self.count_red_lines)

            if self.count_red_lines == 2:
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

                        self.grass_terrain_detected = True
                        break
                    twist = Twist()
                    twist.linear.x = 0
                    twist.angular.z = 0
                    self.cmd_pub.publish(twist)
                    # time.sleep(1)
                    self.start_timer = 0
                    print("Terrain Detected Pause")





        self.start_timer += TIME_STEP
        print(self.start_timer)

    

        


    def clock_callback(self,msg):
        current_time = msg.clock

    def charToStr(charImg):
        char_dict = {
            0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
            9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R",
            18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z",
            26: "0", 27: "1", 28: "2", 29: "3", 30: "4", 31: "5", 32: "6", 33: "7", 34: "8", 35: "9"
        }

        charImg = cv2.resize(charImg, (30, 36))
        charImg = np.expand_dims(charImg, axis=0)
        
        charPred = model_license.predict(charImg)[0]
        index = np.argmax(charPred)

        return char_dict[index]

    def plateToStr(self, posCount, plateImg, w, h):
        #split the plate into characters
        firstCharImg = plateImg[0:h, 0:int(w*1.0/3)]
        secondCharImg = plateImg[0:h, int(w*0.9/3): int(w*0.9/2)]
        thirdCharImg = plateImg[0:h, int(w*1.2/2): int(w*2.3/3)]
        fourthCharImg= plateImg[0:h, int(w*2.3/3): w]
        
        #convert each character image to string
        firstCharStr = self.charToStr(firstCharImg)
        secondCharStr = self.charToStr(secondCharImg)
        thirdCharStr = self.charToStr(thirdCharImg)
        fourthCharStr = self.charToStr(fourthCharImg)

        return "Position " + posCount + ": " + firstCharStr + secondCharStr + thirdCharStr + fourthCharStr


    def processFrame(self, frame):
        height, width = frame.shape[:2]

        cropImage = frame[int(height*0.6):height, 0:width]
        hsvCropImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        hsvCropImage = cv2.bilateralFilter(hsvCropImage,10,100,100)

        # mask and extract the license plate
        mask = cv2.inRange(hsvCropImage, LOWER_HSV, UPPER_HSV)

        maskBin = mask.astype(np.uint8) * 255

        # Count the number of blue pixels in the ROI
        pixelCount = cv2.countNonZero(maskBin)

        return mask, cropImage, pixelCount
    
    def plateDetection(self, frame):
        mask, cropImage, pixelCount = self.processFrame(self, frame)
        if(pixelCount>3300):
                _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
                largestLabel = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                x, y, w, h, _ = stats[largestLabel]
                result = cropImage[y:y+h, x:x+w]
                self.license_pub.publish(String(self.plateToStr(result)))



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
    
    def onehot_to_string(self,index):
        # Define a dictionary mapping indices to characters
        char_dict = {
            0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
            9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R",
            18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z",
            26: "0", 27: "1", 28: "2", 29: "3", 30: "4", 31: "5", 32: "6", 33: "7", 34: "8", 35: "9"
        }

        # Find the index of the maximum value in the one-hot vector
        # Return the corresponding character from the dictionary
        return char_dict[index]
        


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
