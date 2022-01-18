import math
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from sys import platform
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence, to_categorical

GENDER_UNKNOWN = 'unknown'
GENDERS = {
    GENDER_UNKNOWN: 0,
    'male': 1,
    'female': 2
}

AGE_NAN = 0


class ImageMeta(object):
    SHAPE = (2,)

    def __init__(self, gender: str, age: float):
        self._gender = gender
        self._age = age

    @property
    def gender(self) -> str:
        return self._gender

    @property
    def age(self) -> float:
        return self._age

    def encode(self) -> np.ndarray:
        if self._gender not in GENDERS:
            gender = GENDER_UNKNOWN
        else:
            gender = self._gender

        if math.isnan(self._age):
            age = AGE_NAN
        else:
            age = self._age

        meta = np.zeros(shape=self.SHAPE)
        meta[0] = GENDERS[gender]
        meta[1] = age

        return meta


class SequencedDataGenerator(Sequence):

    def __init__(self, sequences):
        self._sequences = sequences

    def on_epoch_end(self):
        for seq in self._sequences:
            if hasattr(seq, 'on_epoch_end'):
                seq.on_epoch_end()

    def __getitem__(self, index):
        for seq in self._sequences:
            if index < len(seq):
                return seq[index]
            index -= len(seq)

        return None

    def __len__(self):
        return sum([len(seq) for seq in self._sequences])


class LabeledDataGenerator(Sequence):

    def __init__(self, data_frame: pd.DataFrame,
                 image_root: Path,
                 classes: List[str],
                 batch_size: int,
                 shuffle: bool,
                 categorical_class_mode: bool = True,
                 image_size: Tuple[int, int] = (244, 244)):
        self._data_frame = data_frame.copy()
        self._image_root = image_root
        self._classes = classes
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._categorical_class_mode = categorical_class_mode
        self._image_size = image_size

    def on_epoch_end(self):
        if self._shuffle:
            self._data_frame = self._data_frame.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        batches = self._data_frame[index * self._batch_size:(index + 1) * self._batch_size]

        path_batch = batches['image_id']
        gender_batch = batches['sex']
        age_batch = batches['age']
        name_batch = batches['label']

        x_image_batch = np.asarray([self._load_image(path)
                                    for path in path_batch])
        x_meta_batch = np.asarray([self._load_meta(gender, age)
                                   for gender, age
                                   in zip(gender_batch, age_batch)])
        y_batch = np.asarray([self._load_label(name, len(self._classes))
                              for name in name_batch])

        assert not np.any(np.isnan(x_image_batch))
        assert not np.any(np.isnan(x_meta_batch))
        assert not np.any(np.isnan(y_batch))

        return [x_image_batch, x_meta_batch], y_batch

    def __len__(self):
        return len(self._data_frame) // self._batch_size

    def _load_meta(self, gender: str, age: float) -> np.ndarray:
        return ImageMeta(gender, age).encode()

    def _load_image(self, image_name: str) -> np.ndarray:
        img = image.load_img(str(self._image_root / f'{image_name}.jpg'),
                             target_size=self._image_size)
        return image.img_to_array(img)

    def _load_label(self, label: str, num_labels: int):
        if self._categorical_class_mode:
            return to_categorical(self._classes.index(label), num_labels)
        else:
            return np.asarray([self._classes.index(label)])


class LabeledImageDataset(ABC):

    @property
    @abstractmethod
    def classes(self) -> List[str]:
        pass

    @abstractmethod
    def split_train_test(self, test_size: float):
        pass

    @abstractmethod
    def create_generator(self,
                         batch_size: int,
                         shuffle: bool = True,
                         categorical_class_mode: bool = True):
        pass


class DataFrameLabeledImageDataset(LabeledImageDataset):

    def __init__(self, data_frame: pd.DataFrame, image_root: Path,
                 image_size: Tuple[int, int],
                 classes: List[str]):
        self._data_frame = data_frame
        self._image_root = image_root
        self._image_size = image_size
        self._classes = classes

    @property
    def classes(self) -> List[str]:
        return self._classes

    def split_train_test(self, test_size: float):
        _, df_val = train_test_split(self._data_frame,
                                     test_size=test_size,
                                     stratify=self._data_frame['label'])

        def identify_val_rows(x):
            # create a list of all the lesion_id's in the val set
            val_list = list(df_val['image_id'])

            if str(x) in val_list:
                return 'val'
            else:
                return 'train'

        # identify train and val rows

        # create a new colum that is a copy of the image_id column
        self._data_frame['train_or_val'] = self._data_frame['image_id']
        # apply the function to this new column
        self._data_frame['train_or_val'] = self._data_frame['train_or_val'].apply(identify_val_rows)
        # filter out train rows
        df_train = self._data_frame[self._data_frame['train_or_val'] == 'train']

        return DataFrameLabeledImageDataset(df_train, self._image_root,
                                            self._image_size,
                                            self.classes), \
               DataFrameLabeledImageDataset(df_val, self._image_root,
                                            self._image_size,
                                            self.classes)

    def create_generator(self,
                         batch_size: int,
                         shuffle: bool = True,
                         categorical_class_mode: bool = True):
        return LabeledDataGenerator(
            self._data_frame,
            self._image_root,
            self.classes,
            batch_size,
            shuffle=shuffle,
            categorical_class_mode=categorical_class_mode,
            image_size=self._image_size
        )

    def categorize_images_into_directory(self, output_root: Path):
        for label in self.classes:
            lbl_dir = output_root / label
            if not lbl_dir.exists():
                lbl_dir.mkdir()

        for index, row in self._data_frame.iterrows():
            fname = row['image_id'] + '.jpg'
            label = row['label']

            lbl_dir = output_root / label
            if platform in ('linux', 'linux2'):
                # create a link instead of copying
                os.symlink(self._image_root / fname, lbl_dir / fname)
            else:
                shutil.copy(self._image_root / fname, lbl_dir / fname)


class SequencedLabeledImageDataset(LabeledImageDataset):

    def __init__(self, datasets: List[LabeledImageDataset], classes: List[str]):
        self._datasets = datasets
        self._classes = classes

    @property
    def classes(self) -> List[str]:
        return self._classes

    def split_train_test(self, test_size: float):
        split_pairs = [
            dataset.split_train_test(test_size)
            for dataset in self._datasets
        ]
        return SequencedLabeledImageDataset([pair[0] for pair in split_pairs], self.classes), \
               SequencedLabeledImageDataset([pair[1] for pair in split_pairs], self.classes)

    def create_generator(self, batch_size: int, shuffle: bool = True, categorical_class_mode: bool = True):
        return SequencedDataGenerator(
            [dataset.create_generator(batch_size, shuffle, categorical_class_mode)
             for dataset in self._datasets]
        )


class DatasetLoader(ABC):

    @abstractmethod
    def load(self) -> LabeledImageDataset:
        pass
