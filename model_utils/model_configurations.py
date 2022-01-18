from pathlib import Path

from tensorflow.keras.losses import hinge, categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD

from ml.datasets import CLASS_INDICES, HAM10000DatasetLoader, ISIC2019TrainingDatasetLoader, AcnedtDatasetLoader
from ml.model import create_model, create_resnet50, create_densenet121
from ml.training import ModelTrainer

LOCAL_DIR = Path(__file__).parent / '.local'
CURRENT_DIR = Path(__file__).parent
IMAGE_SIZE = (244, 244)
IMAGE_SHAPE = IMAGE_SIZE + (3,)

DATASET_LOADERS = {
    'ham10000': HAM10000DatasetLoader(
        LOCAL_DIR / 'datasets' / 'ham10000',
        IMAGE_SIZE),
    'isic2019train': ISIC2019TrainingDatasetLoader(
        LOCAL_DIR / 'datasets' / 'isic2019train',
        IMAGE_SIZE),
    'acnedt': AcnedtDatasetLoader(
        CURRENT_DIR / 'datasets' / 'acnedt',
        IMAGE_SIZE)
}

MODEL_CREATORS = {
    'resnet50': lambda: create_resnet50(
        len(CLASS_INDICES),
        IMAGE_SHAPE),
    'densenet121': lambda: create_densenet121(
        len(CLASS_INDICES),
        IMAGE_SHAPE),
    'ourmodel': lambda: create_model(
        len(CLASS_INDICES),
        IMAGE_SHAPE),
}

MODEL_TRAINERS = {
    'resnet50': lambda: ModelTrainer(
        optimizer=SGD(learning_rate=0.0001, momentum=0.9),
        loss_function=categorical_crossentropy
    ),
    'densenet121': lambda: ModelTrainer(
        optimizer=SGD(learning_rate=0.0001, momentum=0.9),
        loss_function=categorical_crossentropy
    ),
    'ourmodel': lambda: ModelTrainer(
        optimizer=Adam(learning_rate=1e-3),
        loss_function=hinge
    ),
}
