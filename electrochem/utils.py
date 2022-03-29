import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import seaborn as sns
from tqdm.notebook import trange, tqdm
import wandb
from wandb.keras import WandbCallback
from random import seed
from random import randint
from easydict import EasyDict as edict

from keras.layers import Dense, LSTM, Input
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, Callback, LearningRateScheduler
from keras.optimizers import Adam, RMSprop

from waylon_keras_utils import train_model
from waylon_layer_utils import AttentionLSTM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, KFold

# All columns for all classes
from constants import ALL_CD_COLS, ALL_CU_COLS, ALL_PB_COLS, ALL_SW_COLS

# Columns without 50ppb samples and best SW columns
from constants import BEST_SW_COLS_MIN_VAL_LESS_THAN_30, \
                      LEAD_NO_50_COLS, \
                      COPPER_NO_500_COLS, \
                      CADMIUM_NO_50_COLS

# ElectroAugmenter default values
from constants import HORIZONTAL_SYSTEMATIC_SHIFT_STD, \
                      VERTICAL_SYSTEMATIC_SHIFT_STD, \
                      VERTICAL_RANDOM_NOISE_SHIFT_STD, \
                      NOISE_SHIFT_SCALING_FACTOR

from electro_augmenter import ElectroAugmenter


DEFAULT_SEED = 7
DEFAULT_SPLITS = 5

CLASS_NAME_TO_INT = {'Cd': 0, 'Cu': 1, 'Pb': 2, 'Sw': 3}
CLASS_INT_TO_NAME = {i: name for name, i in CLASS_NAME_TO_INT.items()}

sns.set()

DATA_DIR = Path('../data')

"""########## CONFIG ##########"""
def load_default_config(drop_50_ppb_cols=False):
    """
    Convenience function to load the default config and keep notebooks
    clean.

    10/05/21 - this dict gives solid performance for the ATTENTION model.
    It is hard to get consistent performance though (probs due to lack of
    data).

    Expected results:
    Val accuracy - 68%-80% (mostly in the mid-low 70% range).

    You will not need to modify most of these values in your experiments,
    instead reassign them with config.model.num_nodes = 512 for each
    experiment.
    """
    config = edict({
        'wandb' : True, # default True
        'cols_to_use': {
            'Cd': ALL_CD_COLS,
            'Cu': ALL_CU_COLS,
            'Pb': ALL_PB_COLS,
            'Sw': ALL_SW_COLS,
        },
        'data_processing': {
            'shuffle': True, # default True
            'seed' : DEFAULT_SEED,
            'n_splits': DEFAULT_SPLITS,
            'a': -1,
            'b': 1,
            'test_size': 0.2 # Changing this will impact batch_size!
        },
        'electro_aug': {
            'batch_size': 50,
            'horizontal_shift': HORIZONTAL_SYSTEMATIC_SHIFT_STD,
            'vertical_shift': VERTICAL_SYSTEMATIC_SHIFT_STD,
            'noise_shift': VERTICAL_RANDOM_NOISE_SHIFT_STD,
            'noise_shift_scale': NOISE_SHIFT_SCALING_FACTOR,
            'multiplier': 10, # default 10
            'shuffle': True, # default True
            'seed': None, # default None
            'aug_pct': 0.8 # default 0.8
        },
        'model': {
            'model_type': 'ATTENTION', # LSTM
            'num_nodes': 256,
            'num_layers': 1, # 2
            'input_shape': (1, 1002),
            'loss': 'categorical_crossentropy',
            'optimizer': 'adam',
            'lr': 1e-4, # if cutoffs is None, this fixed LR is used
            'metrics': ['accuracy']
        },
        'fit_model': {
            'epochs': 200,
            'batch_size': 35, # Can only change this if you change test_size
            'verbose': 2,
            'shuffle': True,
            'class_weight': None # [1, 2, 1, 0.5]
        },
        'fit_generator_model': {
            'epochs': 100,
            'verbose': 2,
            'shuffle': False # Default False as Augmenter handles shuffling
        },
        'callbacks': {
            # EarlyStopping
            'patience': 10,
            'restore_best_weights': True,
            'baseline': None, # Set to None of there isn't one.
            'min_delta': 0.01,
            'monitor': 'val_accuracy', # default 'val_loss'
            # F1_Score
            'average': 'micro',
            # Learning Rate Scheduler
            'cutoffs' : [(0, 1e-3), (35, 1e-4), (60, 1e-5)],
            # W and B
            'save_model' : False # default True
        },
        'plotting': {
            'start_plotting_epoch': 0
        }
    })

    if drop_50_ppb_cols:
        cols_to_use = {
            'Sw': BEST_SW_COLS_MIN_VAL_LESS_THAN_30,
            'Cu': COPPER_NO_500_COLS,
            'Cd': CADMIUM_NO_50_COLS,
            'Pb': LEAD_NO_50_COLS
            }
        config.cols_to_use = cols_to_use

    return config


"""########## LOAD DATA ##########"""

def load_scale_reshape_X_y(config):
    # Load df
    df = pd.read_csv(DATA_DIR / 'four_class_dataset.csv', index_col=0)

    # Only select parts from data processing
    config_dp = config.data_processing
    # Scale X row-wise
    a = config_dp['a']
    b = config_dp['b']
    df_X = df.iloc[:, :-1]
    df_X_scaled = scale_df_X_to_range(df_X, a, b)
    # Reshape into Keras-readable format
    if config.model['model_type'].upper() == 'LSTM':
        # Enable data reshaping to see if this improves model performance.
        new_shape = list(config.model['input_shape'])
        new_shape.insert(0, -1)
        X = df_X_scaled.values.reshape(new_shape)
    elif config.model['model_type'].upper() == 'ATTENTION':
        new_shape = list(config.model['input_shape'])
        new_shape.insert(0, -1)
        X = df_X_scaled.values.reshape(new_shape)
    else:
        raise ValueError("""Please enter supported config.model['model_type']:
                            LSTM or Attention""")

    # Transform y into Keras-readable format
    y = df.iloc[:, -1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y)
    return X, y


def load_scale_reshape_X_y_ATTENTION(a, b, input_shape):
    # Load df
    df = pd.read_csv(DATA_DIR / 'four_class_dataset.csv', index_col=0)

    # Scale X row-wise
    df_X = df.iloc[:, :-1]
    df_X_scaled = scale_df_X_to_range(df_X, a, b)
    # # Reshape into Keras-readable format
    # new_shape = list(input_shape)
    # new_shape.insert(0, -1)
    # X = df_X_scaled.values.reshape(new_shape)

    X = df_X_scaled.values

    # Transform y into Keras-readable format
    y = df.iloc[:, -1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    # y = to_categorical(y)
    return X, y


def load_X_y_from_columns(config):
    """
    Load in X and y data and select a subset of columns to use.

    Defaults to using all columns.

    config must contain a dict cols_to_use with the following key/value
    pairs:

    cols_to_use = {
        'Sw': LIST_OF_SW_COLS,
        'Cu': LIST_OF_CU_COLS,
        'Cd': LIST_OF_CD_COLS,
        'Pb': LIST_OF_LEAD_COLS
        }

    Suggested use in a notebook (for quick prototyping):

    # Load default config
    >>> config = load_default_config()
    # Create cols_to_use dict
    >>> cols_to_use = {-insert-dict-here-}
    # Add cols_to_use dict to config
    >>> config.cols_to_use = cols_to_use
    # Call train(config) as normal
    >>> history = train(config)
    """
    # Read in dataframes
    cadmium_full = pd.read_csv(DATA_DIR / 'cadmium.csv', index_col=0)
    copper_full = pd.read_csv(DATA_DIR / 'copper.csv', index_col=0)
    lead_full = pd.read_csv(DATA_DIR / 'lead.csv', index_col=0)
    seawater_full = pd.read_csv(DATA_DIR / 'seawater.csv', index_col=0)

    # Select columns to use
    cadmium = cadmium_full.loc[:, config.cols_to_use['Cd']]
    copper = copper_full.loc[:, config.cols_to_use['Cu']]
    lead = lead_full.loc[:, config.cols_to_use['Pb']]
    seawater = seawater_full.loc[:, config.cols_to_use['Sw']]

    # Concatenate and scale
    dfs = [cadmium.T, copper.T, lead.T, seawater.T]
    df = pd.concat(dfs, axis=0)
    df_X_unscaled = df.iloc[:, :-1]

    df_y = df.iloc[:, -1]

    return df_X_unscaled, df_y


def scale_X_y(config, df_X, df_y):
    """
    Scale each row in df_X to be in the range [a, b].
    Note: the min value of df_X is mapped to a and the max is mapped
          to b. Each row is not mapped to [a, b] separately.
    """
    df_X_scaled = scale_df_X_to_range(df_X,
                                      config.data_processing.a,
                                      config.data_processing.b)
    # No need to scale y
    return df_X_scaled, df_y


def reshape_to_keras_format(config, df_X, df_y):
    """
    Reshapes df_X and df_y into a format that Keras will accept.
    X is reshape to be 3D and y is OHE'd
    Returns NumPy arrays.
    """
    if config.model.model_type.upper() == 'ATTENTION':
        X = df_X.values
        X = X.reshape(-1, 1, 1002)
        y_values = df_y.values
        y = to_categorical(y_values, num_classes=4)
        return X, y
    else:
        raise ValueError('Only ATTENTION model type supported')

"""########## SPLIT DATA #############"""

def split_X_y(config, X, y):
    config_dp = config.data_processing
    seed = config_dp.get('seed', DEFAULT_SEED)
    shuffle = config_dp.get('shuffle', True)

    if(shuffle and seed is None):
        # seed supplied explicitly as None, generate random seed
        seed = randint(0, 1024)
        config_dp['seed'] = seed

    X_train, X_val, y_train, y_val = train_test_split(
                                        X, y,
                                        test_size=config_dp['test_size'],
                                        random_state=seed,
                                        shuffle=shuffle,
                                        stratify=y)
    return X_train, X_val, y_train, y_val


"""########## SCALE ##########"""
# Misleading name! This is short for _scale_value (a single value) as opposed
# to a sequence.
def _scale_val(val, a, b, minimum, maximum):
    # Scale val into [a, b] given the max and min values of the seq
    # it belongs to.
    # Taken from this SO answer: https://tinyurl.com/j5rppewr
    numerator = (b - a) * (val - minimum)
    denominator = maximum - minimum
    return (numerator / denominator) + a


# Taken from this SO answer: https://tinyurl.com/j5rppewr
def _scale_to_range(seq, a, b, min=None, max=None):
    """
    Given a sequence of numbers - seq - scale all of its values to the
    range [a, b].

    Default behaviour will map min(seq) to a and max(seq) to b.
    To override this, set max and min yourself.
    """
    assert a < b
    # Default is to use the max of the seq as the min/max
    # Can override this and input custom min and max values
    # if, for example, want to scale to ranges not necesarily included
    # in the data (as in our case with the train and val data)
    if max is None:
        max = max(seq)
    if min is None:
        min = min(seq)
    assert min < max
    scaled_seq = np.array([_scale_val(val, a, b, min, max) \
                           for val in seq])

    return scaled_seq


def scale_df_X_to_range(df_X, a, b):
    """
    Given df_X (a pd.DataFrame containing just X values where each row is a
    time series and each column is a voltage), scale each row to the range [a, b].

    Take the min and max values for the sequence to be -40 and 40 (found by
    inspecting all time series individually). Thus, -40 is mapped to a and 40 is
    mapped to b.

    Returns: df_X_scaled - a pd.DataFrame object identical to df_X but each row is
             scaled as discussed.
    """
    scaled_rows = []
    for row in df_X.itertuples(index=False):
        scaled_row = _scale_to_range(row, a, b, min=-40, max=40)
        scaled_rows.append(scaled_row)
    df_X_scaled = pd.DataFrame(scaled_rows, columns=df_X.columns)
    return df_X_scaled


"""########## ELECTRO AUGMENTER ##########"""

def get_electro_augmenter(config, X, y):
    config_ea = config.electro_aug

    aug = ElectroAugmenter(
        X,
        y,
        batch_size=config_ea.batch_size,
        horizontal_shift=config_ea.horizontal_shift,
        vertical_shift=config_ea.vertical_shift,
        noise_shift=config_ea.noise_shift,
        noise_shift_scale=config_ea.noise_shift_scale,
        multiplier=config_ea.multiplier,
        shuffle=config_ea.shuffle,
        seed=config_ea.seed,
        augmentation_percentage=config_ea.aug_pct)

    return aug


"""########## PLOT ##########"""

def plot_metric(history, metric='loss', ylim=None, start_epoch=0):
    """
    * Given a Keras history, plot the specific metric given.
    * Can also plot '1-metric'
    * Set the y-axis limits with ylim
    * Since you cannot know what the optimal y-axis limits will be ahead of time,
      set the epoch you will start plotting from (start_epoch) to avoid plotting
      the massive spike that these curves usually have at the start and thus rendering
      the plot useless to read.
    """
    # Define here because we need to remove '1-' to calculate the right
    title = f'{metric.title()} - Training and Validation'
    ylabel = f'{metric.title()}'
    is_one_minus_metric = False

    if metric.startswith('1-'):
        # i.e. we calculate and plot 1 - metric rather than just metric
        is_one_minus_metric = True
        metric = metric[2:]
    metric = metric.lower()

    fig, ax = plt.subplots()
    num_epochs_trained = len(history.history[metric])
    epochs = range(1, num_epochs_trained + 1)

    values = history.history[metric]
    val_values = history.history[f'val_{metric}']

    if is_one_minus_metric:
        values = 1 - np.array(history.history[metric])
        val_values = 1 - np.array(history.history[f'val_{metric}'])
    else:
        values = history.history[metric]
        val_values = history.history[f'val_{metric}']

    ax.plot(epochs[start_epoch:], values[start_epoch:], 'b', label='Training')
    ax.plot(epochs[start_epoch:], val_values[start_epoch:], 'r', label='Validation')

    ax.set(title=title,
           xlabel='Epoch',
           ylabel=ylabel,
           ylim=ylim)
    ax.legend()
    plt.show()

    return fig


def line_plot_with_title(data, color='b', title=''):
    plt.plot(data, color)
    plt.title(title)
    plt.show()


def print_summary_stats(X, y_true, y_preds):
    """
    Prints accuracy, number of correct predictions and number of incorrect
    predictions.
    """
    accuracy = len(X[y_preds == y_true]) / len(X)
    num_correct = len(X[y_preds == y_true])
    num_incorrect = len(X[y_preds != y_true])
    print(f'NUM CORRECT: {num_correct}/{len(X)}')
    print(f'NUM INCORRECT: {num_incorrect}/{len(X)}')
    print(f'ACCURACY:', accuracy)


def plot_preds_vs_actuals(model, X, y, plot_kind='all'):
    """
    Plots X data coloured green or red depending on whether the model made a correct prediction or not.

    * model - a trained Keras model
    * X - X data in a shape that Keras will accept e.g. (samples, 1, timesteps) if using an Attention model
    * y - y data in a shape that Keras will accept
    * plot_kind - str default 'all', options are 'all', 'correct', 'incorrect' - specify if you just want to plot
                  correct or incorrect predictions.
    """
    # Get predictions and true values in a shape matplotlib likes
    y_preds = make_predictions(model, X)
    y_true = np.array([np.argmax(ohe_value) for ohe_value in y])

    # Print summary stats
    print_summary_stats(X, y_true, y_preds)

    for i in range(len(X)):
        # Get str names for pred and true values
        pred_name = CLASS_INT_TO_NAME[y_preds[i]]
        true_name = CLASS_INT_TO_NAME[y_true[i]]

        # Color based on in/correct prediction
        if pred_name == true_name:
            color = 'green'
        else:
            color = 'red'

        data_to_plot = X[i].reshape(-1,)
        title = f'Predicted {pred_name} - Actual {true_name}'

        if plot_kind.lower() == 'all':
            line_plot_with_title(data_to_plot, color, title)
        elif plot_kind.lower() == 'incorrect' and color == 'red':
            line_plot_with_title(data_to_plot, color, title)
        elif plot_kind.lower() == 'correct' and color == 'green':
            line_plot_with_title(data_to_plot, color, title)
        else:
            pass


"""########## MODEL BUILD AND FIT ##########"""

def get_opimizer(config):
    config_model = config.model
    if config_model['optimizer'].lower() == 'adam':
        optimizer = Adam(learning_rate=config_model['lr'])
    elif config_model['optimizer'].lower() == 'rmsprop':
        optimizer = RMSprop(learning_rate=config_model['lr'])
    else:
        raise Exception("""Please enter a supported optimizer: Adam or RMSprop.""")
    return optimizer


def build_model(config):
    config_model = config.model
    if config_model['model_type'].upper() == 'LSTM':
        model = build_LSTM(config)
    elif config_model['model_type'].upper() == 'ATTENTION':
        model = build_ATTENTION(config)
    else:
        raise ValueError("config.model['model_type'] must be LSTM or ATTENTION")
    return model


def build_ATTENTION(config):
    config_model = config.model

    layers = [AttentionLSTM(config_model['num_nodes'],
                            input_shape=config_model['input_shape'],
                            return_sequences=True) \
              for _ in range(config_model['num_layers'] - 1)]

    layers.append(AttentionLSTM(config_model['num_nodes'],
                                input_shape=config_model['input_shape'],
                                return_sequences=False))
    layers.append(Dense(4, activation='softmax'))
    model = Sequential(layers)

    optimizer = get_opimizer(config)
    model.compile(loss=config_model['loss'],
                 optimizer=optimizer,
                 metrics=config_model['metrics'])
    return model


def build_LSTM(config):
    config_model = config.model

    model = Sequential([
        LSTM(config_model['num_nodes'], input_shape=config_model['input_shape']),
        Dense(4, activation='softmax')
    ])
    optimizer = get_opimizer(config)
    model.compile(loss=config_model['loss'],
                 optimizer=optimizer,
                 metrics=config_model['metrics'])
    return model

# From this Medium article
# https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
class F1Metric(Callback):

    def __init__(self, average):
        # micro, macro, samples, etc.
        self.average = average

    def on_train_begin(self, logs={}):
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs={}):
        val_pred = self.model.predict(self.model.validation_data[0])
        val_pred = (np.asarray(val_pred)).round()
        val_true = self.model.validation_data[1]
        _val_f1 = f1_score(val_true, val_pred, average=self.average)
        self.val_f1s.append(_val_f1)
        print('- val_f1 %f' % _val_f1)
        return


# Taken from this SO answer https://stackoverflow.com/questions/47676248/accessing-validation-data-within-a-custom-callback
# Validation metrics callback: validation precision, recall and F1
# Some of the code was adapted from https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
class F1Metric2(Callback):

    def __init__(self, average, X_train, y_train):
        super().__init__()
        # micro, macro, samples, etc.
        self.average = average
        self.X_train = X_train
        self.y_train = y_train

    def on_train_begin(self, logs={}):
        self.train_f1s = []
        self.val_f1s = []
        # self.val_recalls = []
        # self.val_precisions = []

    def _make_predictions(self, data):
        pred_proba = self.model.predict(data)
        pred_int = [np.argmax(p) for p in pred_proba]
        preds = to_categorical(pred_int, num_classes=4)
        return preds


    def on_epoch_end(self, epoch, logs):
        # Calc preds on train data
        y_train_pred_proba = self.model.predict(self.X_train)
        y_train_pred_int = [np.argmax(p) for p in y_train_pred_proba]
        y_train_pred = to_categorical(y_train_pred_int, num_classes=4)

        # Calc preds on val data
        X_val = self.validation_data[0]
        y_val = self.validation_data[1]

        y_val_pred_proba = self.model.predict(X_val)
        y_val_pred_int = [np.argmax(p) for p in y_val_pred_proba]
        y_val_pred = to_categorical(y_val_pred_int, num_classes=4)

        # Calc f1_score
        train_f1 = round(f1_score(self.y_train, y_train_pred, average=self.average), 4)
        val_f1 = round(f1_score(y_val, y_val_pred, average=self.average), 4)

        self.train_f1s.append(train_f1)
        self.val_f1s.append(val_f1)

        # Add custom metrics to the logs, so that we can use them with
        # EarlyStop and csvLogger callbacks
        logs[f'f1_{self.average}'] = train_f1
        logs[f'val_f1_{self.average}'] = val_f1

        print(f'— f1_{self.average}: {train_f1} — val_f1_{self.average}: {val_f1}')
        return


def get_callbacks(config, X_train, y_train):
    config_callbacks = config.callbacks
    # EarlyStopping
    es = EarlyStopping(patience=config_callbacks['patience'],
                       restore_best_weights=config_callbacks['restore_best_weights'],
                       baseline=config_callbacks['baseline'],
                       min_delta=config_callbacks['min_delta'],
                       monitor=config_callbacks['monitor'])
    # F1-Micro
    f1_micro = F1Metric2(config_callbacks['average'], X_train, y_train)

    callbacks_list = [es, f1_micro]
    # WandB
    wandb_enable = config.get('wandb', True)
    if(wandb_enable):
        save_model = config_callbacks.get('save_model', True)
        callbacks_list.insert(0, WandbCallback(save_model=save_model))

    # Custom LR scheduler
    if config_callbacks.cutoffs is not None:
        custom_lrs = get_custom_lr_scheduler(config)
        callbacks_list.append(custom_lrs)

    return callbacks_list


def get_aug_callbacks(config):
    """
    Since we do not have X_train or y_train when using ElectroAugmenter,
    we need a new function. It's mostly a copy/paste of get_callbacks but
    without F1Metric2 (since this is the class requiring X_train and y_train).
    """
    config_callbacks = config.callbacks
    # EarlyStopping
    es = EarlyStopping(patience=config_callbacks['patience'],
                       restore_best_weights=config_callbacks['restore_best_weights'],
                       baseline=config_callbacks['baseline'],
                       min_delta=config_callbacks['min_delta'],
                       monitor=config_callbacks['monitor'])

    callbacks_list = [es]
    # WandB
    wandb_enable = config.get('wandb', True)
    if wandb_enable:
        save_model = config_callbacks.get('save_model', True)
        callbacks_list.insert(0, WandbCallback(save_model=save_model))

    # Custom LR scheduler
    if config_callbacks.cutoffs is not None:
        custom_lrs = get_custom_lr_scheduler(config)
        callbacks_list.append(custom_lrs)

    return callbacks_list


def custom_LSTM_lr_scheduler(epoch, lr):
    return 1e-3


def custom_lr_scheduler(cutoffs):

    def lr_scheduler(epoch):
        for cutoff, lr in cutoffs[::-1]:

            # if(epoch == cutoff):
            #     print('Learing Rate: {}'.format(lr))
            if(epoch >= cutoff):
                return lr

    return lr_scheduler


def get_custom_lr_scheduler(config):
    if config.model['model_type'].upper() == 'LSTM':
        lrs = LearningRateScheduler(custom_LSTM_lr_scheduler)
    elif config.model['model_type'].upper() == 'ATTENTION':
        cutoffs = config.callbacks.cutoffs
        scheduler = custom_lr_scheduler(cutoffs)
        lrs = LearningRateScheduler(scheduler)
    else:
        raise ValueError('Only supported model_types are: LSTM or ATTENTION')
    return lrs


def fit_model(config, model, X_train, X_val, y_train, y_val):
    # Get callbacks
    callbacks_list = get_callbacks(config, X_train, y_train)

    config_fit_model = config.fit_model
    history = model.fit(
        X_train,
        y_train,
        epochs=config_fit_model['epochs'],
        batch_size=config_fit_model['batch_size'],
        verbose=config_fit_model['verbose'],
        shuffle=config_fit_model['shuffle'],
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        class_weight=config_fit_model['class_weight']
    )
    return history


def fit_generator_model(config, model, train_seq, X_val, y_val):
    # Get callbacks
    callbacks_list = get_aug_callbacks(config)

    config_fgm = config.fit_generator_model

    history = model.fit_generator(
                train_seq,
                steps_per_epoch=len(train_seq),
                epochs=config_fgm.epochs,
                verbose=config_fgm.verbose,
                validation_data=(X_val, y_val),
                shuffle=config_fgm.shuffle,
                callbacks=callbacks_list
                )
    return history

"""########## MODEL EVALUATE ##########"""
def make_predictions(model, X_data, return_ohe=False, num_classes=4):
    """
    * model - a trained Keras model
    * X_data - the data you want to make predictions on

    Returns array of integer (0, 1, 2, 3) predictions unless return_ohe=True
    then returns predictions in OHE form (with 4 classes by default as
    the original problem was 4 class multiclassification problem).
    """
    pred_proba = model.predict(X_data)
    preds = np.array([np.argmax(p) for p in pred_proba])
    if return_ohe:
        preds = to_categorical(preds, num_classes=num_classes)
    return preds


"""########## WANDB ##########"""
def upload_to_wandb(history, fig_dict):
    """
    * history is a Keras history object
    * fig_dict is a dict where each key is a plot title and each item is
      the matplotlib fig plot itself
    """
    # Turn into df
    history_df = pd.DataFrame.from_dict(history.history)
    # Turn into wandb Table
    history_table = wandb.Table(dataframe=history_df)
    # Log
    wandb.log({'history': history_table})

    for title, fig in fig_dict.items():
        wandb.log({title: wandb.Image(fig)})


"""########## FULL PROCESS ##########"""
def train(config):
    # Load, scale, reshape
    df_X_unscaled, df_y = load_X_y_from_columns(config)
    df_X_scaled, df_y = scale_X_y(config, df_X_unscaled, df_y)
    X, y = reshape_to_keras_format(config, df_X_scaled, df_y)
    # Split
    X_train, X_val, y_train, y_val = split_X_y(config, X, y)

    model = build_model(config)

    wandb_enable = config.get('wandb', True)
    if wandb_enable:
        wandb_run = wandb.init(project='electrochemistry',
                               config=config)

    try:
        history = fit_model(config, model, X_train, X_val, y_train, y_val)
        # Plot loss, accuracy and f1_micro curves
        loss_fig = plot_metric(history, metric='loss', start_epoch=config.plotting['start_plotting_epoch'])
        acc_fig = plot_metric(history, metric='accuracy', start_epoch=config.plotting['start_plotting_epoch'])
        f1_fig = plot_metric(history, metric='f1_micro', start_epoch=config.plotting['start_plotting_epoch'])

        if wandb_enable:
            fig_dict = {'Loss - Training and Validation': loss_fig,
                        'Accuracy - Training and Validation': acc_fig,
                        'F1_Micro - Training and Validation': f1_fig}
            # Store history and figures on wandb
            upload_to_wandb(history, fig_dict)
            wandb_run.finish()
        return history
    finally:
        # Always ensure a run is finished before exiting the program
        if wandb_enable:
            wandb_run.finish()
            print('wandb run finished')


def train_kfold(config):
    # Load data
    X, y = load_scale_reshape_X_y_ATTENTION(a=0, b=1, input_shape=(1, 1002))
    # Get constants for KFold
    config_dp = config.data_processing
    seed = config_dp.get('seed', DEFAULT_SEED)
    shuffle = config_dp.get('shuffle', True)
    n_splits = config_dp.get('n_splits', DEFAULT_SPLITS)

    if(shuffle and seed is None):
        # seed supplied explicitly as None, generate random seed
        seed = randint(0, 1024)
        config_dp['seed'] = seed

    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    fold_number = 1
    for train_idx, val_idx in kfold.split(X, y):
        print(f'FOLD {fold_number}/{n_splits}')
        # Get folds using indices
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Transform into Keras shape
        X_train = X_train.reshape(-1, 1, 1002)
        X_val = X_val.reshape(-1, 1, 1002)
        y_train = to_categorical(y_train, num_classes=4)
        y_val = to_categorical(y_val, num_classes=4)

        # Initialize wandb
        wandb_enable = config.get('wandb', True)
        if wandb_enable:
            wandb_run = wandb.init(project='electrochemistry',
                                   config=config)

        try:
            model = build_model(config)
            history = fit_model(config, model, X_train, X_val, y_train, y_val)
            # Plot loss, accuracy and f1_micro curves
            loss_fig = plot_metric(history, metric='loss', start_epoch=config.plotting['start_plotting_epoch'])
            acc_fig = plot_metric(history, metric='accuracy', start_epoch=config.plotting['start_plotting_epoch'])
            f1_fig = plot_metric(history, metric='f1_micro', start_epoch=config.plotting['start_plotting_epoch'])
            # Upload plots to wandb
            if wandb_enable:
                fig_dict = {'Loss - Training and Validation': loss_fig,
                            'Accuracy - Training and Validation': acc_fig,
                            'F1_Micro - Training and Validation': f1_fig}
                # Store history and figures on wandb
                upload_to_wandb(history, fig_dict)
                wandb_run.finish()

            fold_number += 1
        finally:
            # Always ensure a run is finished before exiting the program
            if wandb_enable:
                wandb_run.finish()
                print('wandb run finished')


def train_with_aug(config):
    # Load, scale
    df_X_unscaled, df_y = load_X_y_from_columns(config)
    df_X_scaled, df_y = scale_X_y(config, df_X_unscaled, df_y)

    X = df_X_scaled.values
    y = to_categorical(df_y.values, num_classes=4)

    # Split
    X_train, X_val, y_train, y_val = split_X_y(config, X, y)
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    # print('X_val shape:', X_val.shape)
    # print()

    aug_train = get_electro_augmenter(config, X_train, y_train)
    aug_val = get_electro_augmenter(config, X_val, y_val)

    model = build_model(config)

    wandb_enable = config.get('wandb', True)
    if wandb_enable:
        wandb_run = wandb.init(project='electrochemistry',
                               config=config)
    try:
        history = model.fit_generator(
            aug_train,
            steps_per_epoch=len(aug_train),
            epochs=100,
            verbose=1,
            validation_data=(X_val, y_val),
            shuffle=False
            )
        # Plot loss, accuracy and f1_micro curves
        loss_fig = plot_metric(history, metric='loss', start_epoch=config.plotting['start_plotting_epoch'])
        acc_fig = plot_metric(history, metric='accuracy', start_epoch=config.plotting['start_plotting_epoch'])
        # Not sure I can calculate F1 since the class requires X_train and y_train
        # f1_fig = plot_metric(history, metric='f1_micro', start_epoch=config.plotting['start_plotting_epoch'])

        if wandb_enable:
            fig_dict = {'Loss - Training and Validation': loss_fig,
                        'Accuracy - Training and Validation': acc_fig}
                        # 'F1_Micro - Training and Validation': f1_fig}
            # Store history and figures on wandb
            upload_to_wandb(history, fig_dict)
            wandb_run.finish()
        return history
    finally:
        # Always ensure a run is finished before exiting the program
        if wandb_enable:
            wandb_run.finish()
            print('wandb run finished')


def train_with_aug_kfold(config):
    # Load, scale
    df_X_unscaled, df_y = load_X_y_from_columns(config)
    df_X_scaled, df_y = scale_X_y(config, df_X_unscaled, df_y)

    X = df_X_scaled.values
    y = to_categorical(df_y.values, num_classes=4)

    # Get constants for KFold
    config_dp = config.data_processing
    seed = config_dp.get('seed', DEFAULT_SEED)
    shuffle = config_dp.get('shuffle', True)
    n_splits = config_dp.get('n_splits', DEFAULT_SPLITS)

    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    # Dict to store the models created
    models = {}

    fold_number = 1
    for train_idx, val_idx in kfold.split(X, y):
        print(f'FOLD {fold_number}/{n_splits}')
        # Get folds using indices
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

        aug_train = get_electro_augmenter(config, X_train, y_train)

        model = build_model(config)

        wandb_enable = config.get('wandb', True)
        if wandb_enable:
            wandb_run = wandb.init(project='electrochemistry',
                                config=config)
        try:
            history = fit_generator_model(config, model, aug_train, X_val, y_val)
            # Plot loss and accuracy curves (no f1 as it only works with X_train/y_train arrays)
            loss_fig = plot_metric(history, metric='loss', start_epoch=config.plotting['start_plotting_epoch'])
            acc_fig = plot_metric(history, metric='accuracy', start_epoch=config.plotting['start_plotting_epoch'])

            if wandb_enable:
                fig_dict = {'Loss - Training and Validation': loss_fig,
                            'Accuracy - Training and Validation': acc_fig}
                # Store history and figures on wandb
                upload_to_wandb(history, fig_dict)
                wandb_run.finish()
            models[f'fold_{fold_number}'] = model
            fold_number += 1
        finally:
            # Always ensure a run is finished before exiting the program
            if wandb_enable:
                wandb_run.finish()
                print('wandb run finished')
    return models


def train_final_model(config):
    """
    After training many models, it's time to make one final model
    trained on all the data.
    """
    # Load, scale
    df_X_unscaled, df_y = load_X_y_from_columns(config)
    df_X_scaled, df_y = scale_X_y(config, df_X_unscaled, df_y)

    X = df_X_scaled.values
    y = to_categorical(df_y.values, num_classes=4)

    aug_all_data = get_electro_augmenter(config, X, y)

    model = build_model(config)

    wandb_enable = config.get('wandb', True)
    if wandb_enable:
        wandb_run = wandb.init(project='electrochemistry',
                               config=config)

    # Get callbacks
    callbacks_list = get_aug_callbacks(config)

    config_fgm = config.fit_generator_model

    history = model.fit_generator(
                aug_all_data,
                steps_per_epoch=len(aug_all_data),
                epochs=config_fgm.epochs,
                verbose=config_fgm.verbose,
                shuffle=config_fgm.shuffle,
                callbacks=callbacks_list
                )
    if wandb_enable:
        wandb_run.finish()
    return history, model