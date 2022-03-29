import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import seaborn as sns
from tqdm.notebook import trange, tqdm
import wandb

sns.set()

DATA_DIR = Path('../data')

# Do not read in 'Mix 1' sheet, as that has been updated in 'mix_1_updated.xlsx'
sheet_names = ['Seawater - No Heavy Metals', 'Copper', 'Cadmium', 'Lead']
xcel = pd.read_excel(DATA_DIR / 'main.xlsx', sheet_name=sheet_names)
# Read in updated mix sheet
mix = pd.read_excel(DATA_DIR / 'mix_1_updated.xlsx')

seawater = xcel['Seawater - No Heavy Metals']
copper = xcel['Copper']
cadmium = xcel['Cadmium']
lead = xcel['Lead']


def get_voltage_series():
    """
    The voltage series runs from 1 to -1 and then from -1 to 1.
    The difference between each voltage is 0.004.
    This series runs along the top of the main.xlsx file and this
    is a convenience function to generate it.
    """
    first_pass = np.arange(1, -1.004, -0.004)
    first_pass[250] = 0
    second_pass = np.arange(-1, 1.004, 0.004)
    second_pass[250] = 0

    voltage_series = np.append(first_pass, second_pass)
    return voltage_series


def create_unique_col_names(col_names):
    """
    Given a list/array of column names, make each unique by appending
    an integer to the end.

    Returns: list (each element being a str)
    """
    i = 0
    cols_unique = []
    for col in col_names:
        col_unique = f'{col}_{i}'
        cols_unique.append(col_unique)
        i += 1
    return cols_unique


def get_longform_df(df):
    # Rename unnamed columns to something readable
    df = df.rename(columns={'Unnamed: 0': 'description',
                            'Unnamed: 1': 'element'})
    # Create new column (we will use this to extract labels later on)
    df['metal_concentration'] = df['element'] + '_' + df['Concentration']
    df = df.drop(columns=['description', 'element', 'Concentration'])
    # Transpose df (now columns are a range - 0, 1, 2, etc.)
    df_T = df.T
    # Select row called metal_concentration (this contains the non-unique col names)
    cols_non_unique = df_T.loc['metal_concentration', :].values
    # Set columns to unique names 
    df_T.columns = create_unique_col_names(cols_non_unique)
    # Drop row with index 'metal_concentration'
    df_T = df_T.drop(index='metal_concentration')

    # Create column 'voltage' (1 to -1 and back to 1)
    df_T['voltage'] = get_voltage_series()

    # Change col order so 'voltage' is at the front
    volt_col_first = np.roll(df_T.columns, 1)
    df_T = df_T.loc[:, volt_col_first]

    # Create a RangeIndex and drop old one
    df_T = df_T.reset_index()
    df_T = df_T.drop('index', axis=1)

    return df_T


def create_wideform_with_unique_names_col(df):
    df_copy = df.copy()
    df_copy = df_copy.rename(columns={'Unnamed: 0': 'description'}, inplace=True)
    df_copy['description'] = df_copy['description'].str[3:]
    df_copy['description'] = df_copy.description.str.replace(' ', '_')
    df_copy.drop(['Analyte', 'Concentration'], axis=1, inplace=True)
    df_copy['unique_names'] = create_unique_col_names(df_copy.description.values)
    df_copy.drop('description', axis=1, inplace=True)
    unique_names_first = np.roll(df_copy.columns, 1)
    df_copy = df_copy.loc[:, unique_names_first]
    return df_copy