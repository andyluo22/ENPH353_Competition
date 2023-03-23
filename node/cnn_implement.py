import csv
import cv2
import numpy as np
import torch
import tensorflow as tf

import csv
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    csv_file = '/home/fizzer/ros_ws/src/controller_pkg/node/commands.csv'
    img_dir = '/home/fizzer/ros_ws/src/controller_pkg/node/'
    data = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            img_filename = row[2]
            img = cv2.imread(img_dir + img_filename)
            cv_imageResize = cv2.resize(img, (224, 224)) # resize the image to 224x224
            im_gray = cv2.cvtColor(cv_imageResize, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(im_gray, 160, 255, cv2.THRESH_BINARY)
            img = binary_image[130:224, :] # crop top half of the image\
            img = img.reshape(img.shape + (1,))
            img = img.astype(np.float32) / 255.0  # scale pixel values to be between 0 and 1
            linear_vel = round(float(row[0]), 2)
            angular_vel = round(float(row[1]), 2)
            data.append((img, [linear_vel, angular_vel]))

    # Split the data into training, validation, and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    img_shape = (94,224,1)  # the shape of your input images
    num_commands = 2 # assuming you have two output commands: linear velocity and angular velocity

    # Define the deep learning model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_commands)
    ])

        # Compile the model
    model.compile(optimizer='adam',
                loss='mse')

    # Convert the data to TensorFlow Datasets
    train_dataset = tf.data.Dataset.from_generator(lambda: train_data,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=((94, 224, 1), (2,)))
    val_dataset = tf.data.Dataset.from_generator(lambda: val_data,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=((94, 224, 1), (2,)))
    test_dataset = tf.data.Dataset.from_generator(lambda: test_data,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=((94, 224, 1), (2,)))

    # Train the model
    model.fit(train_dataset.batch(32),
            validation_data=val_dataset.batch(32),
            epochs=10)
    
    # Save the model
    model.save('my_model')

    # Evaluate the model on the test dataset
    loaded_model = tf.keras.models.load_model('my_model')
    # test_loss = loaded_model.evaluate(test_dataset.batch(32))
    # # Print the test loss
    # print('Test loss:', test_loss)



    # Compile the model
    # model.compile(optimizer='adam',
    #               loss='mse')

    # # Convert the data to TensorFlow Datasets
    # train_dataset = tf.data.Dataset.from_generator(lambda: train_data,
    #                                                output_types=(tf.float32, tf.float32),
    #                                                output_shapes=((94, 224, 1), (2,)))
    # val_dataset = tf.data.Dataset.from_generator(lambda: val_data,
    #                                              output_types=(tf.float32, tf.float32),
    #                                              output_shapes=((94, 224, 1), (2,)))

    # # Train the model
    # model.fit(train_dataset.batch(32),
    #           validation_data=val_dataset.batch(32),
    #           epochs=10)


# -----------------------------------------------

# if __name__ == '__main__':
#     csv_file = '/home/fizzer/ros_ws/src/controller_pkg/node/commands.csv'
#     # img_dir = 'src/controller_pkg/node/'
#     img_dir = '/home/fizzer/ros_ws/src/controller_pkg/node/'

#     with open(csv_file, 'r') as f:
#         reader = csv.reader(f)
#         next(reader)  # skip header row
#         data = []
#         for row in reader:
#             img_filename = row[2]
#             img = cv2.imread(img_dir + img_filename)
#             cv_imageResize = cv2.resize(img, (224, 224)) # resize the image to 224x224
#             _, binary_image = cv2.threshold(cv_imageResize, 120, 255, cv2.THRESH_BINARY)
#             img = binary_image[130:224, :] # crop top half of the image
#             img = img.astype(np.float32) / 255.0  # scale pixel values to be between 0 and 1
#             linear_vel = round(float(row[0]), 2)
#             angular_vel = round(float(row[1]), 2)
#             data.append((img, linear_vel, angular_vel))
#             # data.append((img, linear_vel, 0.9))

            

#     # print(tuple(d[0] for d in data))   
#     images = np.array([d[0] for d in data])
#     linear_velocities = np.array([d[1] for d in data])
#     # print(tuple(d[1] for d in data))   
#     angular_velocities = np.array([d[2] for d in data])
#     # print(tuple(d[2] for d in data))   

#     img_shape = (94,224,3)  # the shape of your input images
#     num_commands = 2 # assuming you have two output commands: linear velocity and angular velocity

#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=img_shape),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
#         tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(num_commands, activation='linear')
#     ])

#     # Train the CNN to imitate the expert
#     model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
#     model.fit(images, (linear_velocities, angular_velocities), batch_size=50, epochs=10)

#     # Save the trained model
#     model_filename = 'imitation_model.h5'
#     model.save(model_filename)

# ------------------------------------------------

# if __name__ == '__main__':
#     csv_file = 'commands.csv'
#     # img_dir = 'src/controller_pkg/node/'
#     img_dir = '/home/fizzer/ros_ws/src/controller_pkg/node/'

#     with open(csv_file, 'r') as f:
#         reader = csv.reader(f)
#         next(reader)  # skip header row
#         data = []
#         for row in reader:
#             img_filename = row[2]
#             img = cv2.imread(img_dir + img_filename)
#             # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB format
#             img = cv2.resize(img, (224, 224))  # resize to 224 x 224 pixels
#             img = img.astype(np.float32) / 255.0  # scale pixel values to be between 0 and 1
#             linear_vel = round(float(row[0]), 2)
#             angular_vel = round(float(row[1]), 1)
#             data.append((img, linear_vel, angular_vel))

#     images = np.array([d[0] for d in data])
#     linear_velocities = np.array([d[1] for d in data])
#     angular_velocities = np.array([d[2] for d in data])

#     # Convert numpy arrays to PyTorch tensors
#     images_tensor = torch.tensor(images.transpose(0, 3, 1, 2))  # convert from NHWC to NCHW format
#     linear_velocities_tensor = torch.tensor(linear_velocities)
#     angular_velocities_tensor = torch.tensor(angular_velocities)

#     img_shape = (3, 224, 224)  # the shape of your input images
#     num_commands = 2  # assuming you have two output commands: linear velocity and angular velocity

#     # Convert PyTorch tensors to NumPy arrays
#     images_np = images_tensor.numpy()
#     linear_velocities_np = linear_velocities_tensor.numpy()
#     angular_velocities_np = angular_velocities_tensor.numpy()

#     # Train the CNN to imitate the expert
#     trainer = CNNTrainer(img_shape=img_shape, num_commands=num_commands)
#     trainer.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
#     trainer.model.fit(images_np, [linear_velocities_np, angular_velocities_np], batch_size=32, epochs=10)

#     # Save the trained model
#     model_filename = 'imitation_model.pt'
#     trainer.model.save(model_filename)
    
    # - ---------------------------------------------------------
    # # Train the CNNs
    # trainer = CNNTrainer(img_shape=img_shape, num_commands=num_commands)
    # trainer.train(images_np, linear_velocities_np, angular_velocities_np, validation_split=0.2)

    # # Save the trained model
    # model_filename = 'my_model.pt4'
    # trainer.model.save(model_filename)
    