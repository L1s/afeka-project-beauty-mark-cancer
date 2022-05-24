# Afeka project 2022 - Cancer Beauty Moles

Skin Cancer Mole Detector - Deep Learning

## About the Model

**Note: Need to write information.**

- Wanted outputs:
    - Skin Cancer
    - Non-Cancer lesion
    - Healthy Skin

## Setting up Local Machine

### Install Dependencies

Install requirements for _python_. Run `pip install -r requirements.txt`


## Local Requirements 

Required files to run the server (after train with all models):
- Trained weights in the following folder :  `<project-folder>\model_utils\.local\weights\`
- Kaggle.json in the following paths as described below (Creation described in _"datasets"_):
  - Windows: `C:\Users\<Windows-username>\.kaggle\kaggle.json`
  - Unix: `~/.kaggle/kaggle.json`

  
### Datasets

Datasets will be automatically downloaded from _Kaggle_. For this function to work, it is necessary to config an API key
which is connected to a _Kaggle_ account:
( register if you didn't !!!)

- Enter your account settings on _Kaggle_ (`https://www.kaggle.com/<username>/account`)
- Go to API settings, click _Create New API Token_
    - this will download a `kaggle.json` file.
- Place this file in the _Kaggle_ home folder:
    - Windows: `C:\Users\<Windows-username>\.kaggle\kaggle.json`
    - Unix: `~/.kaggle/kaggle.json`

## Programs

### [Analyze Model](analyze_model.py)

Runs different analysis actions on models.

```shell
python analyze_model.py {models} [--output-dir=DIR] [--dataset=DATASET] [--confusion-matrix] [--make-plot]
```

For each given model, the program runs a set of analysis actions depending on the options provided:

- Confusion Matrix: runs prediction with the provided dataset. Generates a confusion matrix from the predictions, and
  the labels as set in the dataset. Displays the matrix with _matplotlib_ python lib.
    - If `--confusion-matrix` option is supplied.
    - Dataset used is provided with `--dataset`.
- Generates a plot describing the model and saves it to a file.
    - If `--make-plot` option is supplied.
    - Plot is saved as an image file to the folder specified in `--output-dir`.

For more information about the options, run

```shell
python analyze_model.py --help
```

### [Train and Evaluate](ml_train.py)

Runs training and evaluation on models.

```shell
python ml_train.py {models} [--dataset=DATASET] [--train] [--evaluate-full]
```

For each given model, the program runs a training/evaluation segment depending on the options provided:

- Training: training with the provided dataset. Weights from training are saved to be used again later.
    - If the model doesn't have any stored weights, training will be executed.
    - If `--train` option is supplied, training will be executed. Previous weights are not loaded.
    - If `--retrain` option is supplied, training will be executed. If previous weights are stored, they will be loaded.
    - If `--plot-training-result` is supplied, after training, a plot is generated with results from the training.
    - If `--no-save-weights` is supplied, weights from the training will not be saved.
- Evaluation: evaluation of the model using the provided dataset.
    - If `--evaluate-full` option is supplied

For example: Running the program with:

```shell
python ml_train.py resnet50 --dataset=ham10000 --train
```

Will use `ham10000` dataset to forcibly train the model (overwriting any saved weights).

For more information about the options, run

```shell
python ml_train.py --help
```

### [Server Main](server_app_main.py)

Runs the server application.

```shell
python server_app_main.py
```

The server will be started on the local host on port `5000`, thus the URL is: `http://127.0.0.1:5000/`
so you can `upload image`  and insert the relevant input for `gender` and `age`
