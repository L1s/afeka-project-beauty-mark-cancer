import argparse
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Model
from model_utils.model_functions import run_function_for_models, weights_path_for_model
from model_utils.model_configurations import MODEL_CREATORS, DATASET_LOADERS
from ml.training import ModelTrainer
from ml.analysis import plot_training_results
from ml.dataset import SequencedLabeledImageDataset
from ml.datasets import CLASS_INDICES

"""
Global - EPOCHS for training
"""
EPOCHS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('models', type=str, nargs='+',
                        help=f'Models to run program on. Options: {", ".join(MODEL_CREATORS.keys())}')
    parser.add_argument('--dataset', type=str, default='ham10000', choices=list(DATASET_LOADERS.keys()) + ['all'],
                        help=f'Dataset to use for training and evaluation.')
    parser.add_argument('--retrain', action='store_true',
                        help='Train the ml overwriting any previous training')
    parser.add_argument('--train', action='store_true',
                        help='Train the ml. If training was done before, this will essentially continue training.')
    parser.add_argument('--plot-training-result', action='store_true',
                        help='Show training results as a plot')
    parser.add_argument('--no-save-weights', action='store_true',
                        help='Do not save changes to weights')
    parser.add_argument('--evaluate-full', action='store_true',
                        help='Run evaluation on the ml using the entire dataset')

    return parser.parse_args()


def main():
    args = parse_args()

    should_show_plot = args.plot_training_result

    print('Using dataset', args.dataset)
    if args.dataset == 'all':
        datasets = [loader.load() for loader in DATASET_LOADERS.values()]
        dataset = SequencedLabeledImageDataset(datasets, list(CLASS_INDICES.keys()))
    else:
        dataset = DATASET_LOADERS[args.dataset].load()

    def action(model: Model, model_trainer: ModelTrainer):
        weights_file = weights_path_for_model(model)

        if args.retrain or args.train or not weights_file.exists():
            if not weights_file.exists():
                print('No weights file')
            elif not args.retrain:
                print('Loading saved weights:', weights_file)
                model.load_weights(str(weights_file))

            print('Training...')
            result = model_trainer.train_and_evaluate(model, dataset,
                                                      test_split_size=0.17,
                                                      epochs=EPOCHS)
            print(result)

            if args.plot_training_result:
                plot_training_results(result, EPOCHS)

            if not args.no_save_weights:
                model.save_weights(str(weights_file), overwrite=True)
        else:
            print('Loading saved weights:', weights_file)
            model.load_weights(str(weights_file))

        if args.evaluate_full:
            print('Evaluating full dataset')
            result = model_trainer.evaluate(model, dataset)
            print(result)

    run_function_for_models(args.models, action)

    if should_show_plot:
        plt.show()


if __name__ == '__main__':
    main()
