import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, ZeroPadding2D

class CNNTrainer:
    def __init__(self, img_shape, num_commands):
        self.img_shape = img_shape
        self.num_commands = num_commands
        self.model = self.build_model()

    # def build_model(self):
        # model = Sequential()
        # model.add(Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=self.img_shape))
        # model.add(BatchNormalization())
        # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D((2, 2), padding='same'))
        # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        # model.add(BatchNormalization())
        # model.add(MaxPooling2D((2, 2), padding='same'))
        # model.add(Flatten())
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(self.num_commands, activation='linear'))

    #     # # Compile the model
    #     model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    #     return model

    def build_model(self):
        # Build a CNN model with the same architecture as before
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=self.img_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_commands, activation='linear'))

        # Use mean squared error loss and Adam optimizer
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        def bc_loss(y_true, y_pred):
            # Compute the difference between predicted and expert actions
            error = y_pred - y_true

            # Compute the mean squared error
            mse = tf.reduce_mean(tf.square(error))

            # Scale the error by the norm of the expert actions
            norm = tf.norm(y_true)
            scaled_error = mse / (norm + 1e-8)

            return scaled_error


        # Compile the model with the custom behavioral cloning loss
        model.compile(loss=bc_loss, optimizer=optimizer)

        return model
    
    # def build_model(self):
    #     # Build a CNN model with the same architecture as before
    #     model = Sequential()
    #     model.add(Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=self.img_shape))
    #     model.add(BatchNormalization())
    #     model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D((2, 2)))
    #     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #     model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #     model.add(BatchNormalization())
    #     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D((2, 2), padding='same'))
    #     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #     model.add(BatchNormalization())
    #     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D((2, 2), padding='same'))
    #     model.add(Flatten())
    #     model.add(Dense(256, activation='relu'))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(128, activation='relu'))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(self.num_commands, activation='linear'))

    #     # Use mean squared error loss and Adam optimizer
    #     loss_fn = tf.keras.losses.MeanSquaredError()
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    #     def bc_loss(y_true, y_pred):
    #         # Compute the difference between predicted and expert actions
    #         error = y_pred - y_true

    #         # Compute the mean squared error
    #         mse = tf.reduce_mean(tf.square(error))

    #         # Scale the error by the norm of the expert actions
    #         norm = tf.norm(y_true)
    #         scaled_error = mse / (norm + 1e-8)

    #         return scaled_error


    #     # Compile the model with the custom behavioral cloning loss
    #     model.compile(loss=bc_loss, optimizer=optimizer)

    #     return model

    # def train(self, X_train, linear_velocities, angular_velocities, batch_size=32, epochs=10, validation_split=0.2):
    #     # Concatenate linear_velocities and angular_velocities arrays to create y_train
    #     y_train = np.concatenate((linear_velocities[:, np.newaxis], angular_velocities[:, np.newaxis]), axis=1)
    #     # Train the CNN model on the input data and output commands
    #     self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def train(self, X_train, linear_velocities, angular_velocities, batch_size=48, epochs=15, validation_split=0.2):
        # Concatenate linear_velocities and angular_velocities arrays to create y_train
        y_train = np.concatenate((linear_velocities[:, np.newaxis], angular_velocities[:, np.newaxis]), axis=1)

        # Train the CNN model using the custom loss function
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)


    def predict(self, X):
        # Use the trained CNN model to predict the output commands for new input images
        y_pred = self.model.predict(X)
        return y_pred
    
