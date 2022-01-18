import itertools
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

from ml.training import TrainingResult


def create_confusion_matrix(y_true: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, np.argmax(predictions, axis=1),
                            labels=np.arange(0, stop=num_classes))


def plot_confusion_matrix(confusionmatrix: np.ndarray,
                          classes: List[str],
                          normalize: bool = False,
                          title: str = 'Confusion matrix',
                          colormap=plt.cm.Blues,
                          fig: plt.Figure = None):
    if not fig:
        fig = plt.figure()

    if normalize:
        confusionmatrix = confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis]

    ax = fig.add_subplot()

    im = ax.imshow(confusionmatrix, interpolation='nearest', cmap=colormap)
    ax.set_title(title)

    fig.colorbar(im)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusionmatrix.max() / 2.
    for i, j in itertools.product(range(confusionmatrix.shape[0]), range(confusionmatrix.shape[1])):
        ax.text(j, i, format(confusionmatrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if confusionmatrix[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    fig.tight_layout()


def plot_training_results(result: TrainingResult, epochs: int,
                          figure: plt.Figure = None):
    if not figure:
        figure = plt.figure(figsize=(8, 8))
        figure.suptitle(result.model.name)

    acc = result.history.history['accuracy']
    val_acc = result.history.history['val_accuracy']

    loss = result.history.history['loss']
    val_loss = result.history.history['val_loss']

    epochs_range = range(epochs)

    ax = figure.add_subplot(1, 2, 1)
    ax.plot(epochs_range, acc, label='Training Accuracy')
    ax.plot(epochs_range, val_acc, label='Validation Accuracy')
    ax.legend(loc='lower right')
    ax.set_title('Training and Validation Accuracy')

    ax = figure.add_subplot(1, 2, 2)
    ax.plot(epochs_range, loss, label='Training Loss')
    ax.plot(epochs_range, val_loss, label='Validation Loss')
    ax.legend(loc='upper right')
    ax.set_title('Training and Validation Loss')
