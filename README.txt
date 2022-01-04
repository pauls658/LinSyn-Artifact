# Configure VM with proper number of cores

# Install
copy files over

unzip dreal
run setup command
pip3 install dreal ipython

in auto_lirpa
python setup.py install develop (note that torch==1.6, numpy==1.19.2)

# Reproducing The Paper Results
The hardware used for our experiments was a Intel 2.6 GHz i7-6700 CPU and 32GB RAM. Note that if you have less than 32GB of RAM, some of the scripts for running experiments may crash with out-of-memory. We also provide scripts that will run a subset of experiments which can be run on a machine with only 8GB of RAM. 

**Note**: occassionaly numpy will throw underflow warnings when synthesizing the bounds. These warnings occur during the local optimization, and thus do not effect the soundness of the computed linear bounds.


## Vision Neural Network Experiments
This section explains how to reproduce the verification results for the MNIST and CIFAR neural networks. We provide the trained networks used in our experiments and scripts to run all of the experiments under the auto_lirpa/examples/vision directory. The instructions below assume the user is in this directory.

We provide two scripts for reproducing results: (1) run_verification.sh which will reproduce all of the vision experiment results, and (2) run_verification_artifact.sh that runs a subset of the experiments. The latter script will attempt to verify 10 properties (i.e. prove robustness for 10 images) on each of the 6 vision networks. After running either of these scripts, a directory "experiments" will be created, which has 6 subdirectories -- one for each network -- that contain the verification result for the corresponding network. We also provide a script for reading the results and printing a nicely-formatted table. Below we show the usage, which works with either script. From the auto_lirpa/examples/vision directory, run:

  # run the verification
  bash run_verification_artifact.sh
  # print the results 
  cd experiments
  python ../print_table_stats.py

The expected output is:
bench                     auto acc    auto time    ours acc    ours time    tot prop
----------------------  ----------  -----------  ----------  -----------  ----------
cnn_4layer_geluopenai/         0          29.54        0.7         77.2           10
cnn_4layer_loglog/             0           3.57        0.3         82.15          10
cnn_4layer_swish/              0.2         1.38        0.7         74.53          10
cnn_5layer_geluopenai/         0         117.42        0.14       114.16           7
cnn_5layer_loglog/             1           9.43        1           66.06           4
cnn_5layer_swish/              0           6.84        0.43       111.86           7

### Details on the scripts


## Language model experiments
This section explains how to reproduce the verification results for the LSTM neural networks. We provide the trained networks used in our experiments and scripts to run all of the experiments under the auto_lirpa/examples/language directory. The instructions below assume the user is in this directory.



### Details on the scripts
**A Note on these scripts:** all inputs must have the same sequence length, and the same sequence index(es) must be perturbed for all the inputs.
