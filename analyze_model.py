import argparse
from pathlib import Path
from typing import Tuple

from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import Model

import numpy as np
import matplotlib.pyplot as plt

from model_utils.model_functions import run_function_for_models, weights_path_for_model
from model_utils.model_configurations import MODEL_CREATORS, LOCAL_DIR, DATASET_LOADERS
from ml.analysis import create_confusion_matrix, plot_confusion_matrix
from ml.dataset import LabeledImageDataset
from ml.datasets import CLASS_INDICES
from ml.training import ModelTrainer


def predict_all_from_dataset(model: Model,
                             dataset: LabeledImageDataset,
                             batch_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    generator = dataset.create_generator(batch_size,
                                         shuffle=False,
                                         categorical_class_mode=False)

    predictions = []
    true_classes = []
    for x_batch, y_batch in generator:
        result = model.predict(x_batch)
        predictions.append(result)
        true_classes.append(y_batch)

    return np.concatenate(predictions, axis=0), \
           np.concatenate(true_classes, axis=0)


def action_create_confusion_matrix(model: Model, dataset: LabeledImageDataset):
    predictions, true_labels = predict_all_from_dataset(model, dataset)
    cm = create_confusion_matrix(true_labels, predictions, len(CLASS_INDICES))
    plot_confusion_matrix(cm, CLASS_INDICES.keys())


def action_plot_model(model: Model, output_dir: Path):
    output_path = output_dir / f'{model.name}_plot.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_model(model, to_file=str(output_path))

    print('Saved plot to:', output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('models', type=str, nargs='+',
                        help=f'Models to run program on. Options: {", ".join(MODEL_CREATORS.keys())}')
    parser.add_argument('--output-dir', type=str, default=str(LOCAL_DIR / 'analysis'),
                        help=f'Directory to save analysis data files.')
    parser.add_argument('--dataset', type=str, default='ham10000', choices=DATASET_LOADERS.keys(),
                        help=f'Dataset to use for training and evaluation.')
    parser.add_argument('--make-plot', action='store_true',
                        help='Creates a plot of the model and saves it.')
    parser.add_argument('--confusion-matrix', action='store_true',
                        help='Run prediction on the model with the entire dataset and display a confusion matrix')

    return parser.parse_args()


def main():
    args = parse_args()

    should_load_dataset = args.confusion_matrix
    should_load_weights = args.confusion_matrix
    should_show_plot = args.confusion_matrix

    if should_load_dataset:
        print('Using dataset', args.dataset)
        dataset = DATASET_LOADERS[args.dataset].load()
    else:
        dataset = None

    def action(model: Model, model_trainer: ModelTrainer):
        if should_load_weights:
            weights_file = weights_path_for_model(model)
            if weights_file.exists():
                print('Loading saved weights:', weights_file)
                model.load_weights(str(weights_file))
            else:
                print('No weights file for model')

        if args.make_plot:
            print('Creating model plot')
            action_plot_model(model, Path(args.output_dir))

        if args.confusion_matrix:
            print('Calculating confusion matrix')
            action_create_confusion_matrix(model, dataset)

    run_function_for_models(args.models, action)

    if should_show_plot:
        plt.show()


if __name__ == '__main__':
    main()
