#!/bin/bash

pip install --upgrade pip
pip install jupyter-core --upgrade
pip install jupyterthemes
pip install --upgrade jupyterthemes
jt -t monokai -f roboto -fs 9 -nf sourcesans -nfs 11 -tf sourcesans -tfs 11 -N -kl -T -cursc x -cellw 95%

pip install seaborn sklearn tqdm wandb jupyterlab