# Install
## VM setup
- Download the VM from this link: https://zenodo.org/record/5562597#.YdPMHJHMIzM
- Import the downloaded VM into VirtualBox
- Under the VM's settings configure a shared folder that points to this file's directory. We set /home/tacas22/shared as the mount point on the VM
- Configure the VM with as many cpus as your machine will allow. More cpus will greatly reduce the runtime.
- If you intended to replicate the environment for the paper's evaluation, you will need to configure 8 cpus and 32GB of RAM

## Software installation
- sign in to the VM. user/pass is tacas22/tacas22. then run the installation script:

  cd ~
  bash shared/install.sh
  
The install script two important things: (1) install packages/software and (2) increase the limit on the maximum number of files allowed to be open. The thing (2) is necessary for python's parallelization library. If you see an error in the install script's output from "ulimit", you will not be able to leverage LinSyn's parallelism.

** Note: our implementation depends on the Gurobi LP solver, which requires a license. After running this installation
   script, gurobi will be installed with a restricted license that is only good for 9 days. After it expires, you will need
   to apply for an academic license (which is free) here: https://www.gurobi.com/downloads/end-user-license-agreement-academic. **

# Reproducing The Paper Results
The hardware used for our experiments was a Intel 2.6 GHz i7-6700 CPU that has 8 CPUs and 32GB RAM. Note that if you have less than 32GB of RAM, some of the scripts for running the full set experiments may crash with out-of-memory. However, we also provide scripts that will run a subset of experiments which can be run on a machine with only 8GB of RAM. 

**Note**: occassionaly numpy will throw underflow warnings when synthesizing the bounds. These warnings occur during the local optimization, and thus do not effect the soundness of the computed linear bounds.

## Vision Neural Network Experiments
This section explains how to reproduce the verification results for the MNIST and CIFAR neural networks. We provide the trained networks used in our experiments and scripts to run all of the experiments under the auto_lirpa/examples/vision directory. The instructions below assume the user is in this directory.

We provide two scripts for reproducing results: (1) run_verification.sh which will reproduce all of the vision experiment results, and (2) run_verification_artifact.sh that runs a subset of the experiments. The latter script will attempt to verify 10 properties (i.e. prove robustness for 10 images) on each of the 6 vision networks. After running either of these scripts, a directory "experiments" will be created, which has 6 subdirectories -- one for each network -- that contain the verification result for the corresponding network. We also provide a script for reading the results and printing a nicely-formatted table. Below we show the usage, which works with either script. From the auto_lirpa/examples/vision directory, run:

  # run the verification
  bash run_verification_artifact.sh
  # print the results 
  cd experiments
  python ../print_table_stats.py

This will print a table with auto_lirpa's % of properties verified (i.e. accuracy) and time, LinSyn's (i.e. ours) % verified and time, and the total number of properties attempted. Below is the expected output for an 8 cpu machine:
bench                     auto acc    auto time    ours acc    ours time    tot prop
----------------------  ----------  -----------  ----------  -----------  ----------
cnn_4layer_geluopenai/         0          29.54        0.7         77.2           10
cnn_4layer_loglog/             0           3.57        0.3         82.15          10
cnn_4layer_swish/              0.2         1.38        0.7         74.53          10
cnn_5layer_geluopenai/         0         117.42        0.14       114.16           7
cnn_5layer_loglog/             1           9.43        1           66.06           4
cnn_5layer_swish/              0           6.84        0.43       111.86           7

### Details on the scripts
The above script simply calls the script 

## Language model experiments
This section explains how to reproduce the verification results for the LSTM neural networks. We provide the trained networks used in our experiments and scripts to run all of the experiments under the auto_lirpa/examples/language directory. The instructions below assume the user is in this directory.

We provide two scripts for reproducing results: (1) run_verification.sh which will reproduce all of the vision experiment results, and (2) run_verification_artifact.sh that runs a subset of the experiments. The latter script will attempt to verify 10 properties (i.e. prove robustness for 10 sentences) on each of the 3 LSTMs used in our paper. After running either of these scripts, a directory "experiments" will be created, which has 3 subdirectories -- one for each network -- that contain the verification result for the corresponding network. We also provide a script for reading the results and printing a nicely-formatted table. Below we show the usage, which works with either script. From the auto_lirpa/examples/vision directory, run:

  # run the verification
  bash run_verification_artifact.sh
  # print the results 
  cd experiments
  python ../print_table_stats.py

This will print a table with % of properties verified (i.e. accuracy) and time for each of the tools evaluated in our paper (auto_lirpa, popqorn, and ours). Below is the expected output for an 8 cpu machine:
bench        ours acc    ours time  auto_lirpa acc      auto_lirpa time     popqorn acc         popqorn time
---------  ----------  -----------  ------------------  ------------------  ------------------  ------------------
hardtanh/    0.777778      306.878  -                   -                   -                   -
loglog/      1             381.149  0.2222222089767456  176.03140425682068  -                   -
sigtanh/     0.888889      170.347  0.8888888955116272  12.318811416625977  0.8888888955116272  1076.5625357627869

### Details on the scripts
**A Note on these scripts:** all inputs must have the same sequence length, and the same sequence index(es) must be perturbed for all the inputs.



# Training new models
## Vision models
python3 train.py --device cpu --eps 0 --lr "1e-3" --model cnn_5layer_swish --data CIFAR --save_model cifar_cnn_5layer_swish_eps00_lr1e3

## Language models
python3 train.py --dir models/lstmcustom_sigtanh --arch "custom-no-avg" --num_epochs=10 --model=lstm --lr=1e-3 --dropout=0.5 --train
