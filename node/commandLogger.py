#!/usr/bin/env python

import rospy
import csv
import cv2
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class CommandLogger:
    def __init__(self, filename, image_topic):
        self.file = open(filename, 'w')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['Linear X', 'Angular Z', 'Image Filename'])
        self.bridge = CvBridge()
        self.image_filename = None
        self.image_sub = rospy.Subscriber(image_topic, Image, self.save_image)

        # Create the "images" folder if it doesn't exist
        if not os.path.exists("images"):
            os.makedirs("images")

    def save_image(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(e)
            return
        timestamp = rospy.Time.now().to_sec()
        self.image_filename = f"images/{timestamp}.jpg"
        cv2.imwrite(self.image_filename, cv_image)

    def log_command(self, msg):
        if self.image_filename is None:
            return
        linear_x = round(msg.linear.x, 2)
        angular_z = round(msg.angular.z, 2)
        self.writer.writerow([linear_x, angular_z, self.image_filename])
        self.file.flush()
        self.image_filename = None

if __name__ == '__main__':
    rospy.init_node('command_logger')
    logger = CommandLogger('commands1.csv', '/R1/pi_camera/image_raw')
    rospy.Subscriber('/R1/cmd_vel', Twist, logger.log_command)
    rospy.spin()



# #!/usr/bin/env python

# import rospy
# import csv
# import cv2
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist
# from cv_bridge import CvBridge

# class CommandLogger:
#     def __init__(self, filename, image_topic):
#         self.file = open(filename, 'w')
#         self.writer = csv.writer(self.file)
#         self.writer.writerow(['Timestamp', 'Linear X', 'Angular Z', 'Image Filename'])
#         self.bridge = CvBridge()
#         self.image_filename = None
#         self.image_sub = rospy.Subscriber(image_topic, Image, self.save_image)

#     def save_image(self, msg):
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         except Exception as e:
#             rospy.logerr(e)
#             return
#         timestamp = rospy.Time.now().to_sec()
#         self.image_filename = f"{timestamp}.jpg"
#         cv2.imwrite(self.image_filename, cv_image)

#     def log_command(self, msg):
#         if self.image_filename is None:
#             return
#         timestamp = rospy.Time.now().to_sec()
#         linear_x = msg.linear.x
#         angular_z = msg.angular.z
#         self.writer.writerow([timestamp, linear_x, angular_z, self.image_filename])
#         self.file.flush()
#         self.image_filename = None

# if __name__ == '__main__':
#     rospy.init_node('command_logger')
#     logger = CommandLogger('commands.csv', '/R1/pi_camera/image_raw')
#     rospy.Subscriber('/R1/cmd_vel', Twist, logger.log_command)
#     rospy.spin()