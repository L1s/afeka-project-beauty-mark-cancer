from typing import Callable, Tuple, List

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.applications.densenet import DenseNet121, \
    preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, \
    preprocess_input as resnet50_preprocess_input
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Input, Layer, Concatenate, Average, \
    RandomFourierFeatures

from ml.dataset import ImageMeta
from ml.datasets.common import CLASS_CATEGORIES

""" 
Global - Default_image_shape = (width, height, channels)
"""
_DEFAULT_IMAGE_SHAPE = (None, None, 3)


class PreprocessingLayer(Layer):

    def __init__(self,
                 preprocessing_function: Callable,
                 name: str = None):
        super().__init__(name=name)
        self._preprocessing_function = preprocessing_function

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        return self._preprocessing_function(inputs)


def create_model(num_classes: int, image_shape: Tuple = _DEFAULT_IMAGE_SHAPE):
    "instantiate a Keras tensor - image_input_layer, meta_input_layer"
    image_input_layer = Input(shape=image_shape, name='image_input_layer')
    meta_input_layer = Input(shape=ImageMeta.SHAPE, name='meta_input_layer')

    resnet = _create_resnet50(image_input_layer)
    densenet = _create_densenet121(image_input_layer)

    "average resnet,densenet - It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape)."
    x = Average()([resnet, densenet])

    "concat with meta"
    x = Concatenate(axis=1, name='concat_models')([x, meta_input_layer])

    x = RandomFourierFeatures(output_dim=4096, scale=10.0, kernel_initializer="gaussian")(x)
    x = Dense(num_classes, activation='softmax', name='classification')(x)
    return Model([image_input_layer, meta_input_layer], x, name='ourmodel')


def create_resnet50(num_classes: int, image_shape: Tuple = _DEFAULT_IMAGE_SHAPE) -> Model:
    image_input_layer = Input(shape=image_shape, name='image_input_layer')
    meta_input_layer = Input(shape=ImageMeta.SHAPE, name='meta_input_layer')

    x = _create_resnet50(image_input_layer)
    x = Dense(num_classes, activation='sigmoid', name='classification')(x)

    return Model([image_input_layer, meta_input_layer], x, name='resnet50')


def _create_resnet50(input_layer) -> Model:
    """
    imagenet -> pre-training on ImageNet
    false include top - whether to include the fully-connected layer at the top of the network.
    """
    base_model = ResNet50(
        input_shape=input_layer.shape[1:],
        weights='imagenet',
        include_top=False)
    "Switch the default trainable to false in each layer"
    for layer in base_model.layers:
        layer.trainable = False
    "The preprocess_input function is meant to adequate/adjust your image to the format the model requires."
    preprocessing = PreprocessingLayer(resnet50_preprocess_input,
                                       name='resnet50_preprocess_input')(input_layer)
    x = base_model(preprocessing)
    "Global average pooling operation for spatial data."
    x = GlobalAveragePooling2D(name='resnet50_glbavgpool_out')(x)
    "adding dense layer"
    x = Dense(1024, activation='relu', name='resnet50_dense_out')(x)

    return x


def create_densenet121(num_classes: int, image_shape: Tuple = _DEFAULT_IMAGE_SHAPE) -> Model:
    image_input_layer = Input(shape=image_shape, name='image_input_layer')
    meta_input_layer = Input(shape=ImageMeta.SHAPE, name='meta_input_layer')

    x = _create_densenet121(image_input_layer)
    x = Dense(1024, activation='relu', name='dense_out')(x)
    x = Dense(num_classes, activation='sigmoid', name='classification')(x)

    return Model([image_input_layer, meta_input_layer], x, name='densenet121')


def _create_densenet121(input_layer) -> Model:
    """
    imagenet -> pre-training on ImageNet
    false include top - whether to include the fully-connected layer at the top of the network.
    """
    base_model = DenseNet121(
        input_shape=input_layer.shape[1:],
        weights='imagenet',
        include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    preprocessing = PreprocessingLayer(densenet_preprocess_input,
                                       name='densenet_preprocess_input')(input_layer)
    x = base_model(preprocessing)
    "Global average pooling operation for spatial data."
    x = GlobalAveragePooling2D(name='densenet121_glbavgpool_out')(x)

    return x


def decode_predictions(predictions: np.ndarray, classes: List[str]) -> List[Tuple[float, str, str]]:
    best_indices = predictions.argmax(axis=1)

    classes = [(predictions[array_index, index], classes[index])
               for array_index, index in enumerate(best_indices)]
    return [(value, cl, CLASS_CATEGORIES[cl].name) for value, cl in classes]
