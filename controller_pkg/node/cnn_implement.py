import csv
import cv2
import numpy as np
import torch

from cnn_trainer import CNNTrainer

if __name__ == '__main__':
    csv_file = 'commands.csv'
    # img_dir = 'src/controller_pkg/node/'
    img_dir = '/home/fizzer/ros_ws/src/controller_pkg/node/'

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        data = []
        for row in reader:
            img_filename = row[2]
            img = cv2.imread(img_dir + img_filename)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB format
            img = cv2.resize(img, (224, 224))  # resize to 224 x 224 pixels
            img = img.astype(np.float32) / 255.0  # scale pixel values to be between 0 and 1
            linear_vel = round(float(row[0]), 2)
            angular_vel = round(float(row[1]), 1)
            data.append((img, linear_vel, angular_vel))

    images = np.array([d[0] for d in data])
    linear_velocities = np.array([d[1] for d in data])
    angular_velocities = np.array([d[2] for d in data])

    # Convert numpy arrays to PyTorch tensors
    images_tensor = torch.tensor(images.transpose(0, 3, 1, 2))  # convert from NHWC to NCHW format
    linear_velocities_tensor = torch.tensor(linear_velocities)
    angular_velocities_tensor = torch.tensor(angular_velocities)

    img_shape = (3, 224, 224)  # the shape of your input images
    num_commands = 2  # assuming you have two output commands: linear velocity and angular velocity

    # Convert PyTorch tensors to NumPy arrays
    images_np = images_tensor.numpy()
    linear_velocities_np = linear_velocities_tensor.numpy()
    angular_velocities_np = angular_velocities_tensor.numpy()
    
    # Train the CNN
    trainer = CNNTrainer(img_shape=img_shape, num_commands=num_commands)
    trainer.train(images_np, linear_velocities_np, angular_velocities_np, validation_split=0.2)

    # Save the trained model
    model_filename = 'my_model.pt4'
    trainer.model.save(model_filename)
    
    # img_shape = (3, 224, 224)  # the shape of your input images
    # num_commands = 2  # assuming you have two output commands: linear velocity and angular velocity

    # # Convert PyTorch tensors to NumPy arrays
    # images_np = images_tensor.numpy()
    # linear_velocities_np = linear_velocities_tensor.numpy()
    # angular_velocities_np = angular_velocities_tensor.numpy()

    # # Train the CNN
    # trainer = CNNTrainer(img_shape=img_shape, num_commands=num_commands)
    # trainer.train(images_np, linear_velocities_np, angular_velocities_np, validation_split=0.2)

    # # Save the trained model
    # model_filename = 'my_model.pt5'
    # trainer.model.save(model_filename)