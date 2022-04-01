# Time-Series / Sequence Multiclass Classification - TensorFlow ⌛

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repo contains a write-up for the first stage of a paid project I completed for a client on Upwork. At the time of writing, we are waiting to start stage 2. The client was delighted with the work and left a 5-star review:

<img width="659" alt="Screenshot 2021-09-21 at 02 06 59" src="https://user-images.githubusercontent.com/51246969/134092832-1301eda9-25ff-4d9b-a28b-38d39eb1278b.png">

# The Problem

The client gave us ~200 rows of data to work with. The data was the output from a machine which analysed seawater. An electric current was passed through each seawater sample and output ~1000 different values. The values vary based on the metals the sample contains. The goal was to classify each of these samples into one of four possible classes: cadmium, copper, lead, and seawater (i.e., no metal found) corresponding to the metal which appears most in the sample. This was complicated somewhat by the concentration of each metal (stage 2 of this project aims to predict the concentration of each metal in the sample).

We built a range of LSTM models and eventually found that an attention-based LSTM worked best and it achieved 97% accuracy. Moreover, to deal with the tiny amount of data we had, we performed extensive data augmentation.

# Noteable Notebooks / Files 📕

- [data_plots.ipynb](https://github.com/codeananda/sequence_classification_tensorflow/blob/main/electrochem/data_plots.ipynb) - plots of all samples, plus plots coloured by un/successful model prediction
- [electro_augmenter.py](https://github.com/codeananda/sequence_classification_tensorflow/blob/main/electrochem/electro_augmenter.py) - data augmentor class that lead to a huge gain in performance
- [train_attention_with_augmentation.ipynb](https://github.com/codeananda/sequence_classification_tensorflow/blob/main/electrochem/train_attention_with_augmentation.ipynb) - training of final attention-based model with data augmentation. This model was saved and submitted to the client.
- [utils.py](https://github.com/codeananda/sequence_classification_tensorflow/blob/main/electrochem/utils.py) - main file containing all functions used for model training

# This Repo is a Work-in-Progress 🏗

This portfolio is a work in progress. It probably won't be in perfect condition when you read it. But I hope it gives you an idea of the quality of my work and what I can do.

If you are interested in working together, please reach out via my [Upwork profile](https://www.upwork.com/freelancers/~01153ca9fd0099730e) or email me at: adamdmurphy4 [at] gmail [dot] com

# Notes

I completed this project alongside a Senior Machine Learning Engineer [Waylon Flinn](https://github.com/waylonflinn). Waylon wrote a custom attention-based LSTM model which provided a significant gain in performance over vanialla LSTMs. All code files he wrote have 'waylon_' appended to the start.
