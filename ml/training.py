from collections import namedtuple
from typing import Callable, List

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD, Optimizer
from tensorflow.python.keras import Model

from ml.dataset import LabeledImageDataset

TrainingResult = namedtuple('TrainingResult', 'model,history,loss,metrics')
EvaluationResult = namedtuple('EvaluationResult', 'model,loss,metrics')


class ModelTrainer(object):

    def __init__(self,
                 optimizer: Optimizer = SGD(learning_rate=0.0001, momentum=0.9),
                 loss_function: Callable = categorical_crossentropy):
        self._optimizer = optimizer
        self._loss_function = loss_function

    def train_and_evaluate(self, model: Model, dataset: LabeledImageDataset,
                           epochs: int = 15,
                           test_split_size: float = 0.17,
                           metrics: List = None) -> TrainingResult:
        if metrics is None:
            metrics = ['accuracy']

        model.compile(optimizer=self._optimizer,
                      loss=self._loss_function,
                      metrics=metrics)

        training_set, test_set = dataset.split_train_test(test_size=test_split_size)

        train_generator = training_set.create_generator(batch_size=32)
        test_generator = test_set.create_generator(batch_size=1, shuffle=False)

        history = model.fit(train_generator,
                            epochs=epochs,
                            validation_data=test_generator)

        result = model.evaluate(test_generator)

        return TrainingResult(model, history,
                              *self._make_metrics_result(result, metrics))

    def evaluate(self, model: Model, dataset: LabeledImageDataset,
                 metrics: List = None) -> EvaluationResult:
        if metrics is None:
            metrics = ['accuracy']

        model.compile(optimizer=self._optimizer,
                      loss=self._loss_function,
                      metrics=metrics)

        generator = dataset.create_generator(batch_size=1, shuffle=False)
        result = model.evaluate(generator)

        return EvaluationResult(model, *self._make_metrics_result(result, metrics))

    def _make_metrics_result(self, result, metric_types: List):
        if isinstance(result, tuple) or isinstance(result, list):
            loss, values = result[0], result[1:]

            other_metrics = dict(zip([
                metric_type.__name__ if callable(metric_type) else str(metric_type)
                for metric_type in metric_types],
                values))

            return loss, other_metrics
        else:
            return result,
