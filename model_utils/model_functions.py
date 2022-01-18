from pathlib import Path
from typing import List, Callable

from tensorflow.python.keras import Model

from model_utils.model_configurations import MODEL_CREATORS, LOCAL_DIR, MODEL_TRAINERS
from ml.training import ModelTrainer


def run_function_for_models(model_names: List[str],
                            fnc: Callable[[Model, ModelTrainer], None]):
    for model_name in model_names:
        if model_name not in MODEL_CREATORS:
            raise RuntimeError(f'Unknown model "{model_name}" requested')

        model = MODEL_CREATORS[model_name]()
        model.summary()

        model_trainer = MODEL_TRAINERS[model_name]()

        fnc(model, model_trainer)


def weights_path_for_model(model: Model) -> Path:
    weights_file = LOCAL_DIR / 'weights' / f'model_{model.name}.h5'
    weights_file.parent.mkdir(parents=True, exist_ok=True)

    return weights_file
