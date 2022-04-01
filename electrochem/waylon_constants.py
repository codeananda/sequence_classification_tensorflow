TRAIN_FILES = [
    "../data/ArrowHead_TRAIN",  # 0
    "../data/ArrowHead_EXP_TRAIN",  # 1
    "../data/Worms_TRAIN",  # 2
    "../data/Worms_EXP_TRAIN",  # 3
    "../data/Seawater_EXP_TRAIN",  # 4
    "../data/Seawater2_TRAIN",  # 5 #
    "../data/Seawater0_EXP_TRAIN",  # 6
    "../data/Seawater_4cats_EXP_TRAIN",  # 7
    "../data/Seawater_TRAIN",  # 8  ####
    "../data/Seawater_4cats_TRAIN",  # 9  ####
    "../data/Explosives_TRAIN",  # 10 ####
    "../data/Explosives_3cats_TRAIN",  # 11 ####
    "../data/Explosives_EXP_TRAIN",  # 12
    "../data/Explosives_3cats_EXP_TRAIN",  # 13
    "../data/Seawater_4cats_truncated_TRAIN",  # 14
    "../data/Seawater_new_TRAIN",  # 15
    "../data/vio_3cats_TRAIN",  # 16
    "../data/Seawater_all_TRAIN",  # 17
    "../data/Seawater_all4cats_TRAIN",  # 18
]


TEST_FILES = [
    "../data/ArrowHead_TEST",  # 0
    "../data/ArrowHead_TEST",  # 1
    "../data/Worms_TEST",  # 2
    "../data/Worms_TEST",  # 3
    "../data/Seawater_TEST",  # 4
    "../data/Seawater2_TEST",  # 5
    "../data/Seawater0_TEST",  # 6
    "../data/Seawater_4cats_TEST",  # 7
    "../data/Seawater_TEST",  # 8  ####
    "../data/Seawater_4cats_TEST",  # 9  ####
    "../data/Explosives_TEST",  # 10 ####
    "../data/Explosives_3cats_TEST",  # 11 ####
    "../data/Explosives_EXP_TEST",  # 12
    "../data/Explosives_3cats_EXP_TEST",  # 13
    "../data/Seawater_4cats_truncated_TEST",  # 14
    "../data/Seawater_new_TEST",  # 15
    "../data/vio_3cats_TEST",  # 16
    "../data/Seawater_all_TEST",  # 17
    "../data/Seawater_all4cats_TEST",  # 18
]


MAX_SEQUENCE_LENGTH_LIST = [
    251,  # 0
    251,  # 1
    900,  # 2
    900,  # 3
    1002,  # 4
    1002,  # 5
    1002,  # 6
    1002,  # 7
    1002,  # 8  ####
    1002,  # 9  ####
    1502,  # 10 ####
    1502,  # 11 ####
    1502,  # 12 ####
    1502,  # 13 ####
    400,  # 14
    1002,  # 15
    484,  # 16
    1002,  # 17
    1002,  # 18
]


NB_CLASSES_LIST = [
    3,  # 0
    3,  # 1
    5,  # 2
    5,  # 3
    11,  # 4
    25,  # 5
    11,  # 6
    4,  # 7
    11,  # 8  ####
    4,  # 9  ####
    11,  # 10 ####
    3,  # 11 ####
    11,  # 12 ####
    3,  # 13 ####
    4,  # 14
    11,  # 15
    3,  # 16
    11,  # 17
    4,  # 18
]


X_AXIS_LIST = [
    "timestep",  # 0
    "timestep",  # 1
    "timestep",  # 2
    "timestep",  # 3
    " ",  # 4
    " ",  # 5
    " ",  # 6
    " ",  # 7
    " ",  # 8  ####
    " ",  # 9  ####
    " ",  # 10 ####
    " ",  # 11 ####
    " ",  # 12 ####
    " ",  # 13 ####
    " ",  # 14
    " ",  # 15
    " ",  # 16
    " ",  # 17
    " ",  # 18
]


Y_AXIS_LIST = [
    "value",  # 0
    "value",  # 1
    "value",  # 2
    "value",  # 3
    " ",  # 4
    " ",  # 5
    " ",  # 6
    " ",  # 7
    " ",  # 8  ####
    " ",  # 9  ####
    " ",  # 10 ####
    " ",  # 11 ####
    " ",  # 12 ####
    " ",  # 13 ####
    " ",  # 14
    " ",  # 15
    " ",  # 16
    " ",  # 17
    " ",  # 18
]
