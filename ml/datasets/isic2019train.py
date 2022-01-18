from pathlib import Path
from typing import Tuple

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from ml.dataset import LabeledImageDataset, DatasetLoader, DataFrameLabeledImageDataset
from ml.datasets import CLASS_INDICES


class ISIC2019TrainingDatasetLoader(DatasetLoader):

    def __init__(self, local_dir: Path, image_size: Tuple[int, int]):
        self._local_dir = local_dir
        self._image_size = image_size

    def load(self) -> LabeledImageDataset:
        if not self._local_dir.exists():
            self._local_dir.mkdir(parents=True)
            self._download_isic_with_kaggle(self._local_dir)

        return self._load_isic_from_directory(self._local_dir)

    def _load_isic_from_directory(self, path: Path) -> LabeledImageDataset:
        df = pd.read_csv(str(path / 'train.csv'))

        df = df.rename(columns={
            'image_name': 'image_id',
            'age_approx': 'age',
            'diagnosis': 'label'
        })

        df = df.replace('AK', 'akiec')
        df = df.replace('BCC', 'bcc')
        df = df.replace('MEL', 'mel')
        df = df.replace('BKL', 'bkl')
        df = df.replace('NV', 'nv')
        df = df.replace('DF', 'df')
        df = df.replace('SCC', 'scc')
        df = df.replace('VASC', 'vasc')

        return DataFrameLabeledImageDataset(
            df,
            path / 'train',
            self._image_size,
            classes=list(CLASS_INDICES.keys())
        )

    def _download_isic_with_kaggle(self, dest_dir: Path):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('cdeotte/jpeg-isic2019-256x256',
                                   path=str(dest_dir),
                                   unzip=True)
