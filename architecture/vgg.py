from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

class vgg:
    def buildModel(height, width, depth):
        inputShape=(height, width, depth)

        model = Sequential
        model.add(Conv2D(64, (3,3), activation='relu', input_shape=inputShape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(128, (3, 3), activation='relu', input_shape=inputShape))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), activation='relu', input_shape=inputShape))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', input_shape=inputShape))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', input_shape=inputShape))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1000, activation='softmax'))

        return model
