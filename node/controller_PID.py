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

        self.moving_car_detected = False
        self.car_frame_counter = 0
        self.prev_car_frame = None



    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        except CvBridgeError as e:
            print(e)
        if(self.countTerrain < 1.000001):
            state = np.zeros(10, dtype=np.int)

            copy = np.copy(cv_image)

            # ----------------------------NON-HSV
            gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)

            blured_image = cv2.GaussianBlur(gray, (3,3), 0)

            edges = cv2.Canny(blured_image, 0, 50)

            isolated = self.region(edges)

            row_indexes, col_indexes = np.nonzero(isolated)
            max_col = 0
            min_col = 0

            # Find find left side of edge and right side of edge
            if len(col_indexes) > 0:
                max_col = max(col_indexes)
                min_col = min(col_indexes)

            # ----------------------------HSV
            if self.count_red_lines == 3:
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
                    
                    self.countTerrain = self.countTerrain + 1

        # --------------------------------------- MOVEMENT CONTROL
        if self.start_timer < 5 and self.countTerrain == 0:
            twist = Twist()
            twist.linear.x = 0.14
            if(min_col > 1000):
                min_col = 50
            twist.angular.z = self.pidLeft.computeLeft(min_col)
            self.cmd_pub.publish(twist)

            self.start_timer += TIME_STEP
            print(self.start_timer)

        elif self.countTerrain == 0: 
            twist = Twist() 
            twist.linear.x = 0.30
            twist.angular.z = self.pid.computeRight(max_col)

            #Detect Red Line Process
            img_red = cv_image[600:720, 300: 1100]
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

            if red_line_points > 0 and self.count_red_lines % 2 == 0:
                self.count_red_lines += 1

            if self.start_timer > 14 and self.count_red_lines == 1:
                self.count_red_lines += 1
                self.movement_detected = True

            if self.count_red_lines % 2 == 1 and self.movement_detected == True :
                twist.linear.x = 0.0
                self.cmd_pub.publish(twist)
                img_ped = cv_image[250:500, 350:1000]

                if self.frame_counter % 3 == 0:
                    if self.prev_frame is not None:
                        # subtract the current frame from the previous frame
                        diff = cv2.absdiff(img_ped, self.prev_frame)
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
                    self.prev_frame = img_ped.copy()
                
                # increment the counter
                self.frame_counter += 1
            
            self.cmd_pub.publish(twist)
            

            self.start_timer += TIME_STEP
            print(self.start_timer)

        elif self.countTerrain == 1:
            twist = Twist() 
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            self.countTerrain = self.countTerrain + 0.0000000000001
            time.sleep(0.5)

            self.start_timer += TIME_STEP
            print(self.start_timer)
            self.terrain_timer = self.start_timer
           

        else:
            try:
                # Pass the image through the model
                angular_vel = float(self.telemetry(cv_image, model))
            except Exception as e:
                print("Error predicting: ", str(e))

            if self.time_terrain_tracker < 0.95:
                linear_vel = 0.1925

            elif self.time_terrain_tracker >= 0.95 and self.time_terrain_tracker < 3.2:
                linear_vel = 0.07
                angular_vel = angular_vel * 2.5

            elif self.time_terrain_tracker >=3.2 and self.time_terrain_tracker < 19:
                linear_vel = 0.08

                if not self.is_turning and time.time() - self.command_timer >= 1.0:
                    self.is_turning = True
                    angular_vel = 1.0 * angular_vel
                    self.start_turn_time = time.time()

                if self.is_turning and time.time() - self.command_timer <= 0.050:
                    angular_vel = -0.010
                else:
                    self.is_turning = False
            elif self.time_terrain_tracker >= 19 and self.time_terrain_tracker < 22:
                linear_vel = 0.04
                angular_vel = angular_vel * 2.4
            
            elif self.time_terrain_tracker >= 22 and self.time_terrain_tracker < 25.5:
                linear_vel = 0.10
                angular_vel = angular_vel 
            
            elif self.time_terrain_tracker >= 25.5 and self.time_terrain_tracker < 30.2:
                linear_vel = 0.05
                try:
                    # Pass the image through the model
                    angular_vel = float(self.telemetryLeft(cv_image, modelLeft)) * 2.4
                except Exception as e:
                    print("Error predicting: ", str(e))

            elif self.time_terrain_tracker >= 30.2 and self.moving_car_detected == False:
                linear_vel = 0
                angular_vel = 0

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
                            self.moving_car_detected == True
                            print("Movement detected.")
                        else:
                            self.movement_detected = False
                            print("No movement detected.")
                            
                    # update the previous frame
                    self.prev_car_frame = img_car.copy()
                
                # increment the counter
                self.car_frame_counter += 1

            
            twist = Twist()
            twist.linear.x = linear_vel
            twist.angular.z = angular_vel 
            self.cmd_pub.publish(twist)
            self.time_terrain_tracker += TIME_STEP
            print("Terrain Timer\n")
            print(self.time_terrain_tracker)
            print("\n")
        


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
    main(sys.argv)