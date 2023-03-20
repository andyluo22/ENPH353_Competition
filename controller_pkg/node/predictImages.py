import os
import cv2
import numpy as np
from tensorflow import keras

from cnn_trainer import CNNTrainer

if __name__ == '__main__':
    # Load the trained model
    model = keras.models.load_model('/home/fizzer/ros_ws/src/controller_pkg/node/my_model.pt4')
    

    # Get the list of image files in the folder
    image_files = os.listdir('/home/fizzer/ros_ws/src/controller_pkg/node/images')

    # Loop through all the image files and make predictions
    for image_file in image_files:
        # Load the image
        image = cv2.imread(os.path.join('images', image_file))

        # Preprocess the image
        cv_imageResize = cv2.resize(image, (224, 224)) # resize the image to 224x224
        input_data = np.transpose(cv_imageResize, (2, 0, 1))  # transpose to (3, 224, 224)
        image = input_data.astype(np.float32) / 255.0  # scale pixel values to be between 0 and 1

        # # Show the resized image
        # cv2.imshow('Resized Image', cv_imageResize)
        # cv2.waitKey(0)

        # Make a prediction
        prediction = model.predict(np.array([image]))
        print("predicting:" + str(image_file) + "\n")

        # Make predictions using trained CNN model            
        linear_vel, angular_vel = prediction[0]

        # Use PID control to generate Twist message

        print("Predictions")
        print(linear_vel)
        print(angular_vel)
        
