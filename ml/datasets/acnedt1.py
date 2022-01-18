import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd

from ml.dataset import LabeledImageDataset, DatasetLoader, DataFrameLabeledImageDataset
from ml.datasets import CLASS_INDICES


class AcnedtDatasetLoader(DatasetLoader):

    def __init__(self, local_dir: Path, image_size: Tuple[int, int]):
        self._local_dir = local_dir
        self._image_size = image_size

    def load(self) -> LabeledImageDataset:
        if not self._local_dir.exists():
            self._local_dir.mkdir(parents=True)
        return self._load_from_directory(self._local_dir)

    def _load_from_directory(self, path: Path) -> LabeledImageDataset:
        df = pd.read_csv(str(path / 'metadata.csv'))

        df = df.rename(columns={
            'image_name': 'image_id',
            'diagnosis': 'label'
        })

        df = df.replace('ACK', 'ack')

        return DataFrameLabeledImageDataset(
            df,
            path / 'images',
            self._image_size,
            classes=list(CLASS_INDICES.keys())
        )
