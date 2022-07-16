import math

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import *

from nets.layers import (GroupNormalization, Locations, RegressBoxes, ScaleExp,
                         UpsampleLike)
from nets.resnet import ResNet50


class PriorProbability(keras.initializers.Initializer):
    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        result = np.ones(shape) * -math.log((1 - self.probability) / self.probability)
        return result

class loc_head():
    def __init__(self, pyramid_feature_size=256):
        options = {
            'kernel_size'        : 3,
            'strides'            : 1,
            'padding'            : 'same',
            'kernel_initializer' : keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None),
            'bias_initializer'   : 'zeros'
        }

        self.features = []
        for i in range(4):
            self.features.append(Conv2D(filters=pyramid_feature_size,name='pyramid_regression_{}'.format(i),**options))
            self.features.append(GroupNormalization())
            self.features.append(Activation('relu'))

        self.reg_outputs_conv       = Conv2D(4, name='pyramid_regression', **options)
        self.reg_outputs_reshape    = Reshape((-1, 4), name='pyramid_regression_reshape')
        
    def call(self, inputs):
        outputs = inputs
        for feature in self.features:
            outputs = feature(outputs)
        
        outputs = self.reg_outputs_conv(outputs)
        outputs = self.reg_outputs_reshape(outputs)
        return outputs

class cls_head():
    def __init__(self, num_classes, pyramid_feature_size=256):
        options = {
            'kernel_size'        : 3,
            'strides'            : 1,
            'padding'            : 'same',
            'kernel_initializer' : keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None),
        }

        self.features = []
        for i in range(4):
            self.features.append(Conv2D(filters=pyramid_feature_size, name='pyramid_classification_{}'.format(i), **options))
            self.features.append(GroupNormalization())
            self.features.append(Activation('relu'))

        self.cls_outputs_conv    = Conv2D(filters=num_classes, name='pyramid_classification'.format(), bias_initializer = PriorProbability(probability=0.01),**options)
        self.cls_outputs_reshape = Reshape((-1, num_classes), name='pyramid_classification_reshape')
        self.cls_outputs_sigmoid = Activation('sigmoid', name='pyramid_classification_sigmoid')
        
        self.cet_outputs_conv    = Conv2D(1, name='pyramid_centerness', **options)
        self.cet_outputs_reshape = Reshape((-1, 1), name='pyramid_centerness_reshape')
        self.cet_outputs_sigmoid = Activation('sigmoid', name='pyramid_centerness_sigmoid')
        
    def call(self, inputs):
        outputs = inputs
        for feature in self.features:
            outputs = feature(outputs)
        
        outputs_cls = self.cls_outputs_conv(outputs)
        outputs_cls = self.cls_outputs_reshape(outputs_cls)
        outputs_cls = self.cls_outputs_sigmoid(outputs_cls)

        outputs_cet = self.cet_outputs_conv(outputs)
        outputs_cet = self.cet_outputs_reshape(outputs_cet)
        outputs_cet = self.cet_outputs_sigmoid(outputs_cet)
        return [outputs_cls, outputs_cet]

def FCOS(inputs_shape, num_classes, strides=[8, 16, 32, 64, 128], mode="predict"):
    inputs      = Input(shape=inputs_shape)

    C3, C4, C5  = ResNet50(inputs)
    
    #-------------------------------------#
    #   80, 80, 512 -> 80, 80, 256
    #   40, 40, 1024 -> 40, 40, 256
    #   20, 20, 2048 -> 20, 20, 256
    #-------------------------------------#
    P3           = Conv2D(256, kernel_size=1, strides=1, padding='same', name='C3_reduced', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None))(C3)
    P4           = Conv2D(256, kernel_size=1, strides=1, padding='same', name='C4_reduced', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None))(C4)
    P5           = Conv2D(256, kernel_size=1, strides=1, padding='same', name='C5_reduced', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None))(C5)
    
    #------------------------------------------------#
    #   20, 20, 256 -> 40, 40, 256 -> 40, 40, 256
    #------------------------------------------------#
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, P4])
    P4           = Add(name='P4_merged')([P5_upsampled, P4])
    #------------------------------------------------#
    #   40, 40, 256 -> 80, 80, 256 -> 80, 80, 256
    #------------------------------------------------#
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, P3])
    P3           = Add(name='P3_merged')([P4_upsampled, P3])

    # 80, 80, 256
    P3 = Conv2D(256, kernel_size=3, strides=1, padding='same', name='P3', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None))(P3)
    # 40, 40, 256
    P4 = Conv2D(256, kernel_size=3, strides=1, padding='same', name='P4', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None))(P4)
    # 20, 20, 256
    P5 = Conv2D(256, kernel_size=3, strides=1, padding='same', name='P5', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None))(P5)
    # 10, 10, 256
    P6 = Conv2D(256, kernel_size=3, strides=2, padding='same', name='P6', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None))(P5)
    # 5, 5, 256
    P7 = Activation('relu', name='C6_relu')(P6)
    P7 = Conv2D(256, kernel_size=3, strides=2, padding='same', name='P7', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None))(P7)

    features = [P3, P4, P5, P6, P7]

    regression_model                = loc_head()
    classification_centerness_model = cls_head(num_classes)

    regressions     = []
    classifications = []
    centerness      = []
    for feature in features:
        regression                  = ScaleExp(2)(regression_model.call(feature))
        classification_centerness   = classification_centerness_model.call(feature)
        
        regressions.append(regression)
        classifications.append(classification_centerness[0])
        centerness.append(classification_centerness[1])
        
    regressions     = Concatenate(axis=1, name="regression")(regressions)
    classifications = Concatenate(axis=1, name="classification")(classifications)
    centerness      = Concatenate(axis=1, name="centerness")(centerness)
    
    if mode == "train":
        pyramids = [classifications, centerness, regressions]
        model = keras.models.Model(inputs=inputs, outputs=pyramids, name="FCOS")
        
        locations   = Locations(strides, name='locations')(features)
        boxes       = RegressBoxes(name='boxes')([locations, regressions])

        pyramids    = [classifications, centerness, boxes]
        prediction_model = keras.models.Model(inputs=inputs, outputs=pyramids, name="FCOS_Predict")
        return model, prediction_model
    else:
        locations   = Locations(strides, name='locations')(features)
        boxes       = RegressBoxes(name='boxes')([locations, regressions])

        pyramids    = [classifications, centerness, boxes]
        model       = keras.models.Model(inputs=inputs, outputs=pyramids, name="FCOS")
        return model
