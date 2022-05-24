import random
from collections import namedtuple
from pathlib import Path
from typing import List

import numpy as np
from readerwriterlock import rwlock
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.np_utils import to_categorical

from app.models import db, AnalysisRequest
from model_utils.model_configurations import MODEL_TRAINERS, DATASET_LOADERS
from ml.dataset import ImageMeta, LabeledImageDataset, SequencedLabeledImageDataset
from ml.datasets import CLASS_INDICES
from ml.datasets.common import CLASS_DESCRIPTIONS
from ml.model import create_model, Model, decode_predictions

Analysis = namedtuple('Analysis', 'name,category,value,description')


class UserProvidedDataGenerator(Sequence):

    def __init__(self,
                 items: List[AnalysisRequest],
                 classes: List[str],
                 batch_size: int,
                 shuffle: bool,
                 categorical_class_mode: bool = True):
        self._items = items
        self._classes = classes
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._categorical_class_mode = categorical_class_mode

    def on_epoch_end(self):
        if self._shuffle:
            random.shuffle(self._items)

    def __getitem__(self, index):
        batches = self._items[index * self._batch_size:(index + 1) * self._batch_size]

        x_image_batch = np.asarray([self._load_image(item) for item in batches])
        x_meta_batch = np.asarray([self._load_meta(item) for item in batches])
        y_batch = np.asarray([self._load_label(item, len(self._classes)) for item in batches])

        assert not np.any(np.isnan(x_image_batch))
        assert not np.any(np.isnan(x_meta_batch))
        assert not np.any(np.isnan(y_batch))

        return [x_image_batch, x_meta_batch], y_batch

    def __len__(self):
        return len(self._items)

    def _load_meta(self, item: AnalysisRequest) -> np.ndarray:
        return ImageMeta(item.gender, item.age).encode()

    def _load_image(self, item: AnalysisRequest) -> np.ndarray:
        return np.frombuffer(item.image)

    def _load_label(self, item: AnalysisRequest, num_labels: int):
        if self._categorical_class_mode:
            return to_categorical(self._classes.index(item.result_label), num_labels)
        else:
            return np.asarray([self._classes.index(item.result_label)])


class UserProvidedDataset(LabeledImageDataset):

    def __init__(self, items: List[AnalysisRequest] = None, classes: List[str] = None):
        if items is None:
            items = self._get_all_objects()
        if classes is None:
            classes = list(CLASS_INDICES.keys())
        self._items = items
        self._classes = classes

    @property
    def classes(self) -> List[str]:
        return self._classes

    def split_train_test(self, test_size: float):
        if int(len(self._items) * test_size) < 1:
            return UserProvidedDataset(self._items, self.classes), \
                   UserProvidedDataset([], self.classes)

        items_train, items_test = train_test_split(self._items, test_size=test_size)
        return UserProvidedDataset(items_train, self.classes), \
               UserProvidedDataset(items_test, self.classes)

    def create_generator(self, batch_size: int, shuffle: bool = True, categorical_class_mode: bool = True):
        return UserProvidedDataGenerator(
            self._items,
            self.classes,
            batch_size,
            shuffle=shuffle,
            categorical_class_mode=categorical_class_mode)

    def _get_all_objects(self) -> List[AnalysisRequest]:
        return db.session.query(AnalysisRequest) \
            .filter(AnalysisRequest.confirmed == True) \
            .all()


class Analyser(object):

    def __init__(self, model: Model, weights_file: Path):
        self._model = model
        self._weights_file = weights_file
        self._weights_lock = rwlock.RWLockRead()

    def analyse(self, image: np.ndarray, meta: ImageMeta) -> Analysis:
        lock = self._weights_lock.gen_rlock()
        lock.acquire(blocking=True)
        try:
            image = np.expand_dims(image, axis=0)
            meta = np.expand_dims(meta.encode(), axis=0)

            predictions = self._model.predict([image, meta],
                                              batch_size=1)
            value, name, category = decode_predictions(predictions, list(CLASS_INDICES.keys()))[0]
            description = '' if name not in CLASS_DESCRIPTIONS else CLASS_DESCRIPTIONS[name]
            return Analysis(name, category, value, description)
        finally:
            lock.release()

    def retrain(self):
        lock = self._weights_lock.gen_wlock()
        lock.acquire(blocking=True)
        try:
            datasets = [loader.load() for loader in DATASET_LOADERS.values()]
            datasets.append(UserProvidedDataset())
            dataset = SequencedLabeledImageDataset(datasets, list(CLASS_INDICES.keys()))

            trainer = MODEL_TRAINERS['ourmodel']()
            trainer.train_and_evaluate(self._model, dataset)

            self._model.save_weights(str(self._weights_file))
        finally:
            lock.release()


class AnalyserProvider(object):

    def __init__(self):
        self._model = None
        self._weights_file = None

    def load(self, num_classes: int, weights_file: Path):
        self._model = create_model(num_classes)
        self._model.load_weights(str(weights_file))
        self._weights_file = weights_file

    @property
    def analyser(self) -> Analyser:
        return Analyser(self._model, self._weights_file)


__provider = AnalyserProvider()


def init_analyser(weights_file: Path):
    __provider.load(len(CLASS_INDICES), weights_file)


def get_analyser() -> Analyser:
    return __provider.analyser
