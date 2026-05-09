from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

def build_model():

    model = Sequential()

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=(128, 128, 3)
        )
    )

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(
            64,
            (3, 3),
            activation='relu'
        )
    )

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(
            128,
            (3, 3),
            activation='relu'
        )
    )

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model