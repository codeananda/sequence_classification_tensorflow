#!/bin/bash

# 22/04/21 - TensorFlow only supports Python 3.6-3.8
# 23/04/21 - TensorFlow 2.4.0 uploaded to conda-forge

# Install from conda-forge (this installs the latest version of python compataible with TF)
conda install -y -q -c conda-forge tensorflow tqdm wandb jupyterthemes jupyter_core jupyter_contrib_nbextensions

# Install from default packages
# nbconvert above 5.6.1 leads to loads of error messages
# https://github.com/ipython-contrib/jupyter_contrib_nbextensions/issues/1529
conda install -y -q jupyter matplotlib seaborn pandas scikit-learn nbconvert=5.6.1

# Set dark mode
jt -t monokai -f roboto -fs 9 -nf sourcesans -nfs 11 -tf sourcesans -tfs 11 -N -kl -T -cursc x -cellw 95%
