from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential


class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (width, height, depth)
        ch_dim = '-1'
        if ch_dim == '1':
            input_shape = (depth, width, height)
        model = Sequential()
        model.add(Conv2D(32,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(ch_dim))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(ch_dim))

        model.add(MaxPool2D(32,(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(ch_dim))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(ch_dim))

        model.add(MaxPool2D(32, (2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense())
        model.add(Activation('relu'))
        model.add(BatchNormalization(ch_dim))
        model.add(Dropout(0.25))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model





