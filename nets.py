# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Flatten, MaxPooling2D, Conv2D)
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import Input


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras import Input
from keras.applications import VGG16
from keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam




class RepresentationModels(object):
    def __init__(self,
                 input_shape=(128, 128, 3),
                 weights=None,
                 class_num=5,
                 lr=0.001,):
        assert weights in ['imagenet', None]
        self.weights = weights
        self.input_shape = input_shape
        self.lr = lr
        self.class_num = class_num

    def get_VGG16(self):
        base_model = VGG16(weights=self.weights,
                           include_top=False,
                           input_shape=self.input_shape)

        if self.weights == 'imagenet':
            for layer in base_model.layers:
                layer.trainable = False

        x = base_model.output
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.class_num, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.summary()
        return self._compile_model(model)

    def get_CNN(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(3, 3, padding="same", activation="relu")(inputs)
        x = Conv2D(8, 3, padding="same", activation="relu")(x)
        x = MaxPooling2D()(x)  # 64
        x = Conv2D(16, 3, padding="same", activation="relu")(x)
        x = MaxPooling2D()(x)  # 32
        x = Conv2D(32, 3, padding="same", activation="relu")(x)
        x = Flatten()(x)
        if self.class_num == 1:
            activation = "sigmoid"
        else:
            activation = "softmax"
        outputs = Dense(self.class_num, activation=activation)(x)
        model = Model(inputs, outputs)
        model.summary()
        return self._compile_model(model)

    def _compile_model(self, model):
        optimizer = Adam(lr=self.lr)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
