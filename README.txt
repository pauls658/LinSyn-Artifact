# Install
## VM setup
- Download the VM from this link: https://zenodo.org/record/5562597#.YdPMHJHMIzM
- Import the downloaded VM into VirtualBox
- Under the VM's settings configure a shared folder that points to this file's directory. We set /home/tacas22/shared as the mount point on the VM
- Configure the VM with as many cpus as your machine will allow. More cpus will greatly reduce the runtime.
- If you intended to replicate the environment for the paper's evaluation, you will need to configure 8 cpus and atleast 24GB of RAM

## Software installation
- sign in to the VM. user/pass is tacas22/tacas22. then run the installation script:

  cd ~
  bash shared/install.sh
  # make sure to close the terminal window and open a new one, otherwise the path environment variables won't be right!
  
The install script two important things: (1) install packages/software and (2) increase the limit on the maximum number of files allowed to be open. The thing (2) is necessary for python's parallelization library. If you see an error in the install script's output from "ulimit", you will not be able to leverage LinSyn's parallelism.

** Note: our implementation depends on the Gurobi LP solver, which requires a license. After running this installation
   script, gurobi will be installed with a restricted license that is only good for 9 days. After it expires, you will need
   to apply for an academic license (which is free) here: https://www.gurobi.com/downloads/end-user-license-agreement-academic. **

# Reproducing The Paper Results
The hardware used for our experiments was a Intel 2.6 GHz i7-6700 CPU that has 8 CPUs and 32GB RAM. Note that if you have less than 32GB of RAM, some of the scripts for running the full set experiments may crash with out-of-memory. However, we also provide scripts that will run a subset of experiments which can be run on a machine with only 8GB of RAM. 

**Note**: occassionaly numpy will throw underflow warnings when synthesizing the bounds. These warnings occur during the local optimization, and thus do not effect the soundness of the computed linear bounds.
**Note**: Gurobi seems to hang for ~60-120 seconds on startup in the VM, which causes longer times than are reported in the paper. This may go away if you obtain the academic license, or if you use a bare-metal machine. 

## Vision Neural Network Experiments
This section explains how to reproduce the verification results for the MNIST and CIFAR neural networks. We provide the trained networks used in our experiments and scripts to run all of the experiments under the auto_lirpa/examples/vision directory. The models are under models_from_paper/. The instructions below assume the user is in this directory.

We provide two scripts for reproducing results: (1) run_verification.sh which will reproduce all of the vision experiment results, and (2) run_verification_artifact.sh that runs a subset of the experiments. The latter script will attempt to verify 10 properties (i.e. prove robustness for 10 images) on each of the 6 vision networks. After running either of these scripts, a directory "experiments" will be created, which has 6 subdirectories -- one for each network -- that contain the verification result for the corresponding network. We also provide a script for reading the results and printing a nicely-formatted table. Below we show the usage, which works with either script. From the auto_lirpa/examples/vision directory, run:

  # run the verification
  bash run_verification_artifact.sh
  # print the results 
  cd experiments
  python ../print_table_stats.py

This will print a table with auto_lirpa's % of properties verified (i.e. accuracy) and time, LinSyn's (i.e. ours) % verified and time, and the total number of properties attempted. Below is the expected output for an 8 cpu machine (the accuracies should match, but the times may differ depending on # of cpus):
bench                     auto acc    auto time    ours acc    ours time    tot prop
----------------------  ----------  -----------  ----------  -----------  ----------
cnn_4layer_geluopenai/         0          29.54        0.7         77.2           10
cnn_4layer_loglog/             0           3.57        0.3         82.15          10
cnn_4layer_swish/              0.2         1.38        0.7         74.53          10
cnn_5layer_geluopenai/         0         117.42        0.14       114.16           7
cnn_5layer_loglog/             1           9.43        1           66.06           4
cnn_5layer_swish/              0           6.84        0.43       111.86           7

### Details on the scripts
Both the artifact script and the full experiments script simply call simple_training.py for each architecure, which is the top-level python script for performing verification. Here we explain the command line options using the following example, which runs the verification for 10 robustness problems for the MNIST swish network:
  
  python3 simple_training.py --device cpu --eps 0.031372549 --model cnn_4layer_swish --data MNIST --bound_type CROWN --load models_from_paper/mnist_$model --verify --conv_mode matrix --num_batches 1 --batch_size 10 --parallel --merge_nodes
  
  --device cpu: this option sets the hardware to use. Currently we only tested with cpu, however if you have a gpu, you should omit this option entirely, and the script will attempt to use the gpu automatically.
  --eps 0.031372549: sets the epsilon value as described in Section 5.1 "The Verification Problem" in the paper. We have 8/255 = 0.031372549.
  --model cnn_4layer_swish: specifies the architecture of the neural network that is being verified. This should match the architecture of the checkpoint that is loaded using the --load option (see below). See the Models = {...} dictionary in models/__init__.py for available architectures. If you wish to support a new archtecture, you must add an entry to this dictionary.
  --data MNIST: the dataset for training/testing. Available options are MNIST and CIFAR.
  --bound_type CROWN: The type of bounds to use. To use LinSyn, you must set this to CROWN.
  --load models_from_paper/mnist_cnn_4layer_swish: the location of the model to load. These models are created by using the training script (see below section on training new models).
  --verify: tells the script that we are only doing verification, and not training. (This script also supports training a model from scratch, see below section on trianing new models).
  --conv_mode matrix: auto_lirpa supports two modes for handling convolution layers: "patches" and "matrix". The former is more efficient than the latter, but auto_lirpa throws an error in this mode for the neural net architectures we use.
  --batch_size 10: the number of robustness problems per batch (i.e. number of images to prove robustness for per batch). Solving multiple problems in batch provides a drastic boost in efficiency (in terms of runtime). It is best to choose the largest possible batch size that will fit into your machine's memory. 
  --num_batches 1: the number of batches to run. The total number of robustness problems will be num_batches*batch_size
  --parallel: wether to enable cpu-level parallelesim. Works for both LinSyn and auto_lirpa.
  --merge_nodes: Specififies that we should use LinSyn to bound the activations. If this flag is ommitted, auto_lirpa's default "decomposing" method will be used to bound the activations.

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
