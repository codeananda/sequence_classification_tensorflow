ALL_CD_COLS = [
    'Cd_50_ppb_0', 'Cd_50_ppb_1', 'Cd_50_ppb_2', 'Cd_50_ppb_3',
       'Cd_125_ppb_4', 'Cd_125_ppb_5', 'Cd_125_ppb_6', 'Cd_125_ppb_7',
       'Cd_250_ppb_8', 'Cd_250_ppb_9', 'Cd_250_ppb_10', 'Cd_250_ppb_11',
       'Cd_375_ppb_12', 'Cd_375_ppb_13', 'Cd_375_ppb_14', 'Cd_375_ppb_15',
       'Cd_500_ppb_16', 'Cd_500_ppb_17', 'Cd_500_ppb_18', 'Cd_500_ppb_19',
       'Cd_625_ppb_20', 'Cd_625_ppb_21', 'Cd_625_ppb_22', 'Cd_625_ppb_23',
       'Cd_750_ppb_24', 'Cd_750_ppb_25', 'Cd_750_ppb_26', 'Cd_750_ppb_27',
       'Cd825_ppb_28', 'Cd825_ppb_29', 'Cd825_ppb_30', 'Cd825_ppb_31',
       'Cd_1000_ppb_32', 'Cd_1000_ppb_33', 'Cd_1000_ppb_34', 'Cd_1000_ppb_35'
       ]

ALL_CU_COLS = [
    'Cu_500_ppb_0', 'Cu_500_ppb_1', 'Cu_500_ppb_2',
       'Cu_500_ppb_3', 'Cu_500_ppb_4', 'Cu_500_ppb_5', 'Cu_500_ppb_6',
       'Cu_500_ppb_7', 'Cu_1000_ppb_8', 'Cu_1000_ppb_9', 'Cu_1000_ppb_10',
       'Cu_1000_ppb_11', 'Cu_1000_ppb_12', 'Cu_1000_ppb_13', 'Cu_1000_ppb_14',
       'Cu_2000_ppb_15', 'Cu_2000_ppb_16', 'Cu_2000_ppb_17', 'Cu_2000_ppb_18',
       'Cu_2000_ppb_19', 'Cu_2000_ppb_20', 'Cu_3000_ppb_21', 'Cu_3000_ppb_22'
       ]

ALL_PB_COLS = [
    'Pd_50_ppb_0', 'Pd_50_ppb_1', 'Pd_50_ppb_2', 'Pd_125_ppb_3',
       'Pd_125_ppb_4', 'Pd_125_ppb_5', 'Pd_125_ppb_6', 'Pd_250_ppb_7',
       'Pd_250_ppb_8', 'Pd_250_ppb_9', 'Pd_250_ppb_10', 'Pd_325_ppb_11',
       'Pd_325_ppb_12', 'Pd_325_ppb_13', 'Pd_325_ppb_14', 'Pd_500_ppb_15',
       'Pd_500_ppb_16', 'Pd_500_ppb_17', 'Pd_500_ppb_18', 'Pd_625_ppb_19',
       'Pd_625_ppb_20', 'Pd_625_ppb_21', 'Pd_625_ppb_22', 'Pd_1000_ppb_23',
       'Pd_1000_ppb_24', 'Pd_1000_ppb_25', 'Pd_1000_ppb_26', 'Pd_750_ppb_27',
       'Pd_750_ppb_28', 'Pd_750_ppb_29', 'Pd_750_ppb_30', 'Pd_875_ppb_31',
       'Pd_875_ppb_32', 'Pd_875_ppb_33', 'Pd_875_ppb_34', 'Pd_50_ppb_35'
       ]

ALL_SW_COLS = [
    'SW0_0', 'SW0_1', 'SW0_2', 'SW0_3', 'SW0_4', 'SW0_5',
       'SW0_6', 'SW0_7', 'SW0_8', 'SW0_9', 'SW0_10', 'SW0_11', 'SW0_12',
       'SW0_13', 'SW0_14', 'SW0_15', 'SW0_16', 'SW0_17', 'SW0_18', 'SW0_19',
       'SW0_20', 'SW0_21', 'SW0_22', 'SW0_23', 'SW0_24', 'SW0_25', 'SW0_26',
       'SW0_27', 'SW0_28', 'SW0_29', 'SW0_30', 'SW0_31', 'SW0_32', 'SW0_33',
       'SW0_34', 'SW0_35', 'SW0_36', 'SW0_37', 'SW0_38', 'SW0_39', 'SW0_40',
       'SW0_41', 'SW0_42', 'SW0_43', 'SW0_44', 'SW0_45', 'SW0_46', 'SW0_47',
       'SW0_48', 'SW0_49', 'SW0_50', 'SW0_51', 'SW0_52', 'SW0_53', 'SW0_54',
       'SW0_55', 'SW0_56', 'SW0_57', 'SW0_58', 'SW0_59', 'SW0_60', 'SW0_61',
       'SW0_62', 'SW0_63', 'SW0_64', 'SW0_65', 'SW0_66', 'SW0_67', 'SW0_68',
       'SW0_69', 'SW0_70', 'SW0_71', 'SW0_72', 'SW0_73', 'SW0_74', 'SW0_75',
       'SW0_76', 'SW0_77', 'SW0_78', 'SW0_79'
        ]

# Columns from seawater.csv with a min_value < -30 that all look flat
# i.e. no random bumps.
# There are 30 columns
BEST_SW_COLS_MIN_VAL_LESS_THAN_30 = [
    'SW0_72', 'SW0_24', 'SW0_48', 'SW0_22',
    'SW0_75', 'SW0_25', 'SW0_47', 'SW0_58',
    'SW0_27', 'SW0_78', 'SW0_21', 'SW0_74',
    'SW0_31', 'SW0_63', 'SW0_29', 'SW0_15',
    'SW0_44', 'SW0_33', 'SW0_46', 'SW0_45',
    'SW0_60', 'SW0_20', 'SW0_77', 'SW0_64',
    'SW0_71', 'SW0_19', 'SW0_52', 'SW0_68',
    'SW0_54', 'SW0_43']

# Copper 500 ppb corresponds to 50 ppb in the lead/cadmium
COPPER_NO_500_COLS = [
    'Cu_1000_ppb_8',
    'Cu_1000_ppb_9',
    'Cu_1000_ppb_10',
    'Cu_1000_ppb_11',
    'Cu_1000_ppb_12',
    'Cu_1000_ppb_13',
    'Cu_1000_ppb_14',
    'Cu_2000_ppb_15',
    'Cu_2000_ppb_16',
    'Cu_2000_ppb_17',
    'Cu_2000_ppb_18',
    'Cu_2000_ppb_19',
    # 'Cu_2000_ppb_20', this has a double peak, so think it makes sense to leave it out.
    'Cu_3000_ppb_21',
    'Cu_3000_ppb_22']

CADMIUM_NO_50_COLS = [
    'Cd_125_ppb_4',
    'Cd_125_ppb_5',
    'Cd_125_ppb_6',
    'Cd_125_ppb_7',
    'Cd_250_ppb_8',
    'Cd_250_ppb_9',
    'Cd_250_ppb_10',
    'Cd_250_ppb_11',
    'Cd_375_ppb_12',
    'Cd_375_ppb_13',
    'Cd_375_ppb_14',
    'Cd_375_ppb_15',
    'Cd_500_ppb_16',
    'Cd_500_ppb_17',
    'Cd_500_ppb_18',
    'Cd_500_ppb_19',
    'Cd_625_ppb_20',
    'Cd_625_ppb_21',
    'Cd_625_ppb_22',
    'Cd_625_ppb_23',
    'Cd_750_ppb_24',
    'Cd_750_ppb_25',
    'Cd_750_ppb_26',
    'Cd_750_ppb_27',
    'Cd825_ppb_28',
    'Cd825_ppb_29',
    'Cd825_ppb_30',
    'Cd825_ppb_31',
    'Cd_1000_ppb_32',
    'Cd_1000_ppb_33',
    'Cd_1000_ppb_34',
    'Cd_1000_ppb_35']

LEAD_NO_50_COLS = [
    'Pd_125_ppb_3',
    'Pd_125_ppb_4',
    'Pd_125_ppb_5',
    'Pd_125_ppb_6',
    'Pd_250_ppb_7',
    'Pd_250_ppb_8',
    'Pd_250_ppb_9',
    'Pd_250_ppb_10',
    'Pd_325_ppb_11',
    'Pd_325_ppb_12',
    'Pd_325_ppb_13',
    'Pd_325_ppb_14',
    'Pd_500_ppb_15',
    'Pd_500_ppb_16',
    'Pd_500_ppb_17',
    'Pd_500_ppb_18',
    'Pd_625_ppb_19',
    'Pd_625_ppb_20',
    'Pd_625_ppb_21',
    'Pd_625_ppb_22',
    'Pd_1000_ppb_23',
    'Pd_1000_ppb_24',
    'Pd_1000_ppb_25',
    'Pd_1000_ppb_26',
    'Pd_750_ppb_27',
    'Pd_750_ppb_28',
    'Pd_750_ppb_29',
    'Pd_750_ppb_30',
    'Pd_875_ppb_31',
    'Pd_875_ppb_32',
    'Pd_875_ppb_33',
    'Pd_875_ppb_34']

CLASS_TO_LABEL = {
    'Cd': 0,
    'Cu': 1,
    'Pb': 2,
    'Sw': 3
    }

LABEL_TO_CLASS = {
    0: 'Cd',
    1: 'Cu',
    2: 'Pb',
    3: 'Sw'
    }

# Calculated in augmentation.ipynb under Shift Everything Left/Right
# Looked at the std of the index of the peaks for Cd, Cu, and Pb.
# The first two were 3.something and the last was 0.9.
# Averaged out, we get 2.
HORIZONTAL_SYSTEMATIC_SHIFT_STD = 2

# The std of the mean for the best seawater samples: BEST_SW_COLS_MIN_VAL_LESS_THAN_30
# Multuply by 0.1 as this is calculated on unscaled data
# Via inspection, 0.1 produces realistic looking results
VERTICAL_SYSTEMATIC_SHIFT_STD = 0.1582920938558759 * 0.1

# The mean std for the first 200 voltages of the best seawater samples
# The first 200 voltages are quite flat (unlike the peaks) and so have
# a more stable std.
VERTICAL_RANDOM_NOISE_SHIFT_STD = 0.21887254718166166

# Use this to control random noise shift.
# By inspection, 0.02 gives a good balance between aug and realistic.
NOISE_SHIFT_SCALING_FACTOR = 0.02