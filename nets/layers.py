
from turtle import shape
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import InputSpec, Layer


class UpsampleLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return tf.compat.v1.image.resize_images(source, (target_shape[1], target_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

class ScaleExp(keras.layers.Layer):
    def __init__(self, init_value, **kwargs):
        super(ScaleExp, self).__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.w = self.add_weight(shape=[1], name=self.name, initializer=initializers.Constant(self.init_value), trainable=True)

    def call(self, inputs, **kwargs):
        x = K.exp(inputs * self.w)
        return x

class Locations(keras.layers.Layer):
    def __init__(self, strides, *args, **kwargs):
        self.strides = strides
        super(Locations, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        feature_shapes          = [K.shape(feature)[1:3] for feature in inputs]
        locations_per_feature   = []
        #--------------------------------------#
        #   对网格进行循环
        #--------------------------------------#
        for feature_shape, stride in zip(feature_shapes, self.strides):
            h = feature_shape[0]
            w = feature_shape[1]

            shifts_x = K.arange(0, w * stride, step=stride, dtype=np.float32)
            shifts_y = K.arange(0, h * stride, step=stride, dtype=np.float32)

            #--------------------------------------#
            #   创建网格
            #--------------------------------------#
            shift_x, shift_y = tf.meshgrid(shifts_x, shifts_y)
            shift_x     = K.reshape(shift_x, (-1,))
            shift_y     = K.reshape(shift_y, (-1,))
            locations   = K.stack((shift_x, shift_y), axis=1) + stride // 2
            locations_per_feature.append(locations)
            
        #--------------------------------------#
        #   网格堆叠
        #--------------------------------------#
        locations = K.concatenate(locations_per_feature, axis=0)
        locations = K.tile(K.expand_dims(locations, axis=0), (K.shape(inputs[0])[0], 1, 1))
        return locations

    def compute_output_shape(self, input_shapes):
        feature_shapes = [feature_shape[1:3] for feature_shape in input_shapes]
        total = 1
        for feature_shape in feature_shapes:
            if None not in feature_shape:
                total = total * feature_shape[0] * feature_shape[1]
            else:
                return input_shapes[0][0], None, 2
        return input_shapes[0][0], total, 2

    def get_config(self):
        config = super(Locations, self).get_config()
        config.update({
            'strides': self.strides,
        })
        return config

class RegressBoxes(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        locations, regression = inputs
        x1 = locations[:, :, 0] - regression[:, :, 0]
        y1 = locations[:, :, 1] - regression[:, :, 1]
        x2 = locations[:, :, 0] + regression[:, :, 2]
        y2 = locations[:, :, 1] + regression[:, :, 3]
        # (batch_size, num_locations, 4)
        bboxes = K.stack([x1, y1, x2, y2], axis=-1)
        return bboxes

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()

        return config

def moments(x, axes, shift=None, keep_dims=False):
    ''' Wrapper over tensorflow backend call '''

    return tf.nn.moments(x, axes, shift=shift, keepdims=keep_dims)

class GroupNormalization(Layer):
    """Group normalization layer.

    Group Normalization divides the channels into groups and computes
    within each group
    the mean and variance for normalization.
    Group Normalization's computation is independent
     of batch sizes, and its accuracy is stable in a wide range of batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical to
    Layer Normalization.

    Relation to Instance Normalization:
    If the number of groups is set to the
    input dimension (number of groups is equal
    to number of channels), then this operation becomes
    identical to Instance Normalization.

    # Arguments
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        mean, variance = moments(inputs, group_reduction_axes[2:],
                                    keep_dims=True)
        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)

        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        # finally we reshape the output back to the input shape
        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
