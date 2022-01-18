import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from ml.dataset import LabeledImageDataset, DatasetLoader, DataFrameLabeledImageDataset
from ml.datasets import CLASS_INDICES


class HAM10000DatasetLoader(DatasetLoader):

    def __init__(self, local_dir: Path, image_size: Tuple[int, int]):
        self._local_dir = local_dir
        self._image_size = image_size

    def load(self) -> LabeledImageDataset:
        if not self._local_dir.exists():
            self._local_dir.mkdir(parents=True)
            self._download_dataset_with_kaggle(self._local_dir)

        self._prepare_ham10000_directory(self._local_dir)
        return self._load_ham10000_from_directory(self._local_dir)

    def _load_ham10000_from_directory(self, path: Path) -> LabeledImageDataset:
        df_data = pd.read_csv(str(path / 'HAM10000_metadata.csv'))

        df = df_data.groupby('lesion_id').count()
        df = df[df['image_id'] == 1]
        df.reset_index(inplace=True)

        def identify_duplicates(x):
            unique_list = list(df['lesion_id'])

            if x in unique_list:
                return 'no_duplicates'
            else:
                return 'has_duplicates'

        # create a new colum that is a copy of the lesion_id column
        df_data['duplicates'] = df_data['lesion_id']
        # apply the function to this new column
        df_data['duplicates'] = df_data['duplicates'].apply(identify_duplicates)
        # now we filter out images that don't have duplicates
        df = df_data[df_data['duplicates'] == 'no_duplicates']

        df = df.rename(columns={'dx': 'label'})

        return DataFrameLabeledImageDataset(
            df,
            path / 'images',
            self._image_size,
            classes=list(CLASS_INDICES.keys())
        )

    def _prepare_ham10000_directory(self, directory: Path):
        image_dir = directory / 'images'
        image_dir.mkdir(parents=True, exist_ok=True)

        for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            if not (directory / folder).exists():
                continue

            for path in (directory / folder).iterdir():
                shutil.move(str(path), str(image_dir))

    def _download_dataset_with_kaggle(self, dest_dir: Path):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('kmader/skin-cancer-mnist-ham10000',
                                   path=str(dest_dir),
                                   unzip=True)
