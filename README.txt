This file contains documentation for reproducing results from the paper, and documentation for the code itself.
It has three sections: Install, Reproducing the Paper Results, and Code Documentation.

# Install
## VM setup
- Download the VM from this link: https://zenodo.org/record/5562597#.YdPMHJHMIzM
- Import the downloaded VM into VirtualBox
- Under the VM's settings configure a shared folder that points to this file's directory. We set /home/tacas22/shared as the mount point on the VM
- Configure the VM with as many cpus as your machine will allow. More cpus will greatly reduce the runtime.
- If you intended to replicate the environment for the paper's evaluation, you will need to configure 8 cpus and atleast 24GB of RAM

## Software installation
- sign in to the VM. user/pass is tacas22/tacas22. Then open a terminal and run the installation script:

  cd ~
  bash shared/install.sh
  # make sure to close the terminal window and open a new one after the script finishes, otherwise the path environment variables won't be right!
  
The install script two important things: (1) install packages/software and (2) increase the limit on the maximum number of files allowed to be open. The thing (2) is necessary for python's parallelization library. If you see an error in the install script's output from "ulimit", you will not be able to leverage thread parallelism.

# Reproducing The Paper Results
The hardware used for our experiments was a Intel 2.6 GHz i7-6700 CPU that has 8 CPUs and 32GB RAM. Note that if you have less than 24GB of RAM, some of the scripts for running the full set experiments may crash with out-of-memory. However, we also provide scripts that will run a subset of experiments which can be run on a machine with only 8GB of RAM and 1 CPU. 

**Important Things To Note**
  - Our implementation depends on the Gurobi LP solver, which requires a license. After running our installation script, gurobi will be installed with a restricted license that is only good for 9 days and also places limits on the size of the LP program that it will solve. The restricted license will only allow you to run the Vision Model experiments, but not the language model experiments. To run all the experiments, You will need to obtain and install an academic license (which is free) here: https://www.gurobi.com/downloads/end-user-license-agreement-academic.
  - occassionaly numpy will throw underflow warnings when synthesizing the bounds. These warnings occur during the local optimization, and thus do not effect the soundness of the computed linear bounds.
  - Gurobi seems to hang for when calling the solver, which causes longer times than are reported in the paper (usually adds about 100 or so seconds to the total runtime). This may go away if you obtain the academic license, or if you use a bare-metal machine. We are not sure why this is happening.

## Vision Neural Network Experiments
This section explains how to reproduce the verification results for the MNIST and CIFAR neural networks. We provide the trained networks used in our experiments and scripts to run all of the experiments under the auto_lirpa/examples/vision directory. The instructions below assume the user is in this directory. The models are under ~/auto_lirpa/examples/vision/models_from_paper/.

We provide two scripts for reproducing results: (1) run_verification.sh which will reproduce all of the vision experiment results, and (2) run_verification_artifact.sh that runs a subset of the experiments. The latter script will attempt to verify 10 properties (i.e. prove robustness for 10 images) on each of the 6 vision networks. After running either of these scripts, a directory "experiments" will be created, which has 6 subdirectories -- one for each network -- that contain the verification result for the corresponding network. We also provide a script for reading the results and printing a nicely-formatted table. Below we show the usage, which works with either script. From the auto_lirpa/examples/vision directory, run:

  # run the verification
  bash run_verification_artifact.sh
  # print the results 
  cd experiments
  python ../print_table_stats.py

This will print a table with auto_lirpa's % of properties verified (i.e. accuracy) and time, LinSyn's (i.e. ours) % verified and time, and the total number of properties attempted. Below is the expected output after running run_verification_artifact.sh on 8 cpu machine (the accuracies should match, but the times may differ depending on # of cpus):
bench                     auto acc    auto time    ours acc    ours time    tot prop
----------------------  ----------  -----------  ----------  -----------  ----------
cnn_4layer_geluopenai/         0          29.54        0.7         77.2           10
cnn_4layer_loglog/             0           3.57        0.3         82.15          10
cnn_4layer_swish/              0.2         1.38        0.7         74.53          10
cnn_5layer_geluopenai/         0         117.42        0.14       114.16           7
cnn_5layer_loglog/             1           9.43        1           66.06           4
cnn_5layer_swish/              0           6.84        0.43       111.86           7

### Details on the scripts
Both the artifact script and the full experiments script simply call simple_training.py for each architecure, which is the top-level python script for performing verification. Here we explain the command line options using the example command below. It takes a trained network (a 4-layer Swish CNN in this case), then chooses 10 images, and attempts to prove robustness of each of those images under a bounded perturbation:
  
  python3 simple_training.py --device cpu --eps 0.031372549 --model cnn_4layer_swish --data MNIST --bound_type CROWN --load models_from_paper/mnist_$model --verify --conv_mode matrix --num_batches 1 --batch_size 10 --parallel --merge_nodes
  
  --device cpu: this option sets the hardware to use. Currently we only tested with cpu, however if you have a gpu, you should omit this option entirely, and the script will attempt to use the gpu automatically.
  --eps 0.031372549: sets the epsilon value as described in Section 5.1 "The Verification Problem" in the paper. We have 8/255 = 0.031372549.
  --model cnn_4layer_swish: specifies the architecture of the neural network that is being verified. This should match the architecture of the checkpoint that is loaded using the --load option (see below). See the Models = {...} dictionary in models/__init__.py for available architectures. If you wish to support a new archtecture, you must add an entry to this dictionary.
  --data MNIST: the dataset for training/testing. Available options are MNIST and CIFAR.
  --bound_type CROWN: The type of bounds to use. To use LinSyn, you must set this to CROWN. See the Auto_LiRPA documentation for more info (https://auto-lirpa.readthedocs.io/en/latest).
  --load models_from_paper/mnist_cnn_4layer_swish: the saved model checkpoint to evaluate robustness on. These models are created by using the training script (see below section on training new models).
  --verify: tells the script that we are only doing verification, and not training. (This script also supports training a model from scratch, see below section on trianing new models).
  --conv_mode matrix: auto_lirpa supports two modes for handling convolution layers: "patches" and "matrix". The former is more efficient than the latter, but auto_lirpa throws an error in this mode for the neural net architectures we use.
  --batch_size 10: the number of robustness problems per batch (i.e. number of images to prove robustness for per batch). Solving multiple problems in batch provides a drastic boost in efficiency (in terms of runtime) for both auto_lirpa's default method and LinSyn. It is best to choose the largest possible batch size that will fit into your machine's memory. 
  --num_batches 1: the number of batches to run. The total number of robustness problems will be num_batches*batch_size
  --parallel: wether to enable cpu-level parallelesim. Works for both LinSyn and auto_lirpa.
  --merge_nodes: Specififies that we should use LinSyn to bound the activations. If this flag is ommitted, auto_lirpa's default "decomposing" method will be used to bound the activations.

## Language model experiments
** Note: ** You will need to obtain and install an academic license for Gurobi for these experiments to work. You can do so here
    https://www.gurobi.com/downloads/end-user-license-agreement-academic.
    
This section explains how to reproduce the verification results for the LSTM neural networks. We provide the trained networks used in our experiments and scripts to run all of the experiments under the auto_lirpa/examples/language directory. The instructions below assume the user is in this directory. The models are under ~/auto_lirpa/examples/language/models_from_paper/.

We provide two scripts for reproducing results: (1) run_verification.sh which will reproduce all of the vision experiment results, and (2) run_verification_artifact.sh that runs a subset of the experiments. The latter script will attempt to verify 10 properties (i.e. prove robustness for 10 sentences) on each of the 3 LSTMs used in our paper. After running either of these scripts, a directory "experiments" will be created, which has 3 subdirectories -- one for each network -- that contain the verification result for the corresponding network. We also provide a script for reading the results and printing a nicely-formatted table. Below we show the usage, which works with either script. From the auto_lirpa/examples/vision directory, run:

  cd ~/auto_lirpa/examples/language
  # run the verification
  bash run_verification_artifact.sh
  # print the results 
  cd experiments
  python ../print_table_stats.py

This will print a table with % of properties verified (i.e. accuracy) and time for each of the tools evaluated in our paper (auto_lirpa, popqorn, and ours). Below is the expected output after running run_verification_artifact.sh on 8 cpu machine (the accuracies should match, but the times may differ depending on # of cpus):
bench        ours acc    ours time  auto_lirpa acc      auto_lirpa time     popqorn acc         popqorn time
---------  ----------  -----------  ------------------  ------------------  ------------------  ------------------
hardtanh/    0.777778      306.878  -                   -                   -                   -
loglog/      1             381.149  0.2222222089767456  176.03140425682068  -                   -
sigtanh/     0.888889      170.347  0.8888888955116272  12.318811416625977  0.8888888955116272  1076.5625357627869

### Details on the scripts
Both the artifact script and the full experiments script simply call run_single_verification.py for each LSTM architecure, which is the top-level python script for performing verification. Here we explain the command line options using the example command below. It takes a trained network (a 1-layer LSTM with sigmoid*tanh activation), then chooses 10 input sequences (i.e. 10 sentences) of length 12, and attempts to that applying a bounded perturbation to the first word's embedding does not change the classification.

  python run_single_verification.py --eps 0.04 --device cpu --method backward --batch_size 10 --max_sent_length 12 --min_sent_length 12 --load models_from_paper/lstm_custom_sigtanh/ckpt_10 --perturb_i 0 --parallel --merge_nodes --tool ours
  
  --eps 0.04: sets the epsilon value as described in Section 5.1 "The Verification Problem" in the paper.
  --device cpu: this option sets the hardware to use. Currently we only tested with cpu, however if you have a gpu, you should omit this option entirely, and the script will attempt to use the gpu automatically.
  --method backward: the bounding method for auto_lirpa to use. Should always be set to backward for our experiments. See the auto_lirpa documentation for more info (https://auto-lirpa.readthedocs.io/en/latest/)
  --batch_size 10: the number of robustness problems per batch (i.e. num of input sentences to prove robustness for). Solving multiple problems in batch provides a drastic boost in efficiency (in terms of runtime) for both auto_lirpa's default method and LinSyn. It is best to choose the largest possible batch size that will fit into your machine's memory.
  --max_sent_length 12: max length of input sequence (i.e. sentence). max and min length should be equal for this script
  --min_sent_length 12: min length of input sequence (i.e. sentence). max and min length should be equal for this script
  --load models_from_paper/lstm_custom_sigtanh/ckpt_10: the saved model checkpoint to evaluate robustness on. These models are created by using the training script (see below section on training new models).
  --perturb_i 0: The index of the input to perturb. In the paper we perturb the first input in the input sequence.
  --parallel: wether to enable cpu-level parallelesim. Works for both LinSyn and auto_lirpa. Has no effect for popqorn.
  --merge_nodes: This flag should be enabled when you wish to use LinSyn or POPQORN's bounding techniques (you specify which one using the --tool option below)
  --tool ours: Which bounding technique to use. Should be one of "ours", "auto_lirpa", or "popqorn". If "auto_lirpa" you need to omit the --merge_nodes flag. If either of the other two, you must include the --merge_nodes flag.

**A Note on these scripts:** all inputs must have the same sequence length, and the same sequence index(es) must be perturbed for all the inputs.

# Training new models
In this section, we show how to train models from scratch.
## Vision models
From the ~/auto_lirpa/examples/vision directory, one can run the following command to train a 5-layer Swish CNN on the CIFAR dataset from scratch:

  python3 simple_training.py --device cpu --eps 0 --num_epochs 100 --lr "1e-3" --model cnn_4layer_swish --data CIFAR --save_model cifar_cnn_5layer_swish
  
Running the above will train and save the model to the file "cifar_cnn_5layer_swish" in the current directory. This file can then be used in the verification examples above. The argument to --model can be any string that is a key in the Models = {...} dictionary in models/__init__.py. The models we used in our experiments are cnn_4layer_swish, cnn_4layer_geluopenai, cnn_4layer_loglog, cnn_5layer_swish, cnn_5layer_geluopenai, cnn_5layer_loglog. For more info on the parameters, one can run python3 simple_training.py --help.

## Language models
From the ~/auto_lirpa/examples/vision directory, one can run the following command to train a 1-layer LSTM with the sigmoid*tanh activation the SST2 dataset from scratch:

  python3 train.py --dir models/lstmcustom_sigtanh --arch "custom-no-avg" --num_epochs=10 --model=lstm --lr=1e-3 --dropout=0.5 --train
  
Running the above will train and save the model to the file under the directory ./models/lstmcustom_sigtanh. The script will actually save the current model after each epoch, so there will be 10 models under the directory ./models/lstmcustom_sigtanh, with the final model being ckpt_10. Any of these checkpoints can then be used in the verification examples above. The argument to --arch specifies the activation pattern to use, which can be one of custom-loglog, custom-hardtanh, custom-sigtanh.


# Code Documentation
Our code has two main modules which are: (1) The LinSyn module located in ~/runtimesyn, and (2) the auto_lirpa framework located in ~/auto_lirpa. 

## LinSyn Module
This section documents the usage of the LinSyn module. The code for this module is located under the directory ~/runtimesyn. The LinSyn algorithm is implemented by the class RuntimeActBounds in the file ~/runtimesyn/RuntimeActBounds.py. This directory also has another file
called ~/runtimesyn/functions.py, which contains the defintions of activation functions used to instantiate a RuntimeActBounds module.

Here we give an example of instantiating a RuntimeActBounds module that will compute bounds for the activation sigmoid(x)*tanh(y). This code is available in ~/runtimesyn/linsynExample.py.

The first step is to write the definition of the activation function. Specifically, we must write a function that returns a dReal expression of the activation function when it is called with dReal Variable arguments, or returns the concrete value of the activation when called with concrete values. For concrete values, the function needs to return numpy floats. Below give an example:

  import dreal as dr, numpy as np
  import torch, math
  drTypes = [dr.Variable, dr.Expression, dr.Variables, dr.Interval, dr.Formula]
  
  # define helper functions first
  def sigmoid(x):
    if type(x) in drTypes: 
        # if argument is a dReal type, return a dReal expression
        return 1 / (1 + dr.exp(-x))
    else: 
        # otherwise return the concrete value as a numpy float
        return 1 / (1 + np.exp(-x))

  def tanh(x):
      if type(x) in drTypes:
          return dr.tanh(x)
      else:
          # here we use the torch implementation of tanh because
          # the implementations between torch and numpy differ, 
          # then we convert the torch float to a numpy float
          return torch.tanh(torch.tensor([x])).numpy()[0]
          
  def sigmoid_tanh(x, y):
    return sigmoid(x)*tanh(y)

Next we need to define the partial derivatives of each variable so that we can compute the jacobian. If you don't know the derivatives off the top of you head, you can use https://www.wolframalpha.com/ to get it. Note that these partial derivative functions are only ever called with concrete values, so we don't actually need to compute the symbolic partial derivatives. You could use pytorch's jacobian function (https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html) to do get the jacobian values.
  
  # Another helper function
  def sech(x):
    if type(x) in drTypes:
        return 1/dr.cosh(x)
    else:
        return 1/np.cosh(x)

  # derivate wrt x
  def dx_sigmoid_tanh(x,y):
    return (1 - sigmoid(x))*sigmoid(x)*tanh(y)
  
  # derivate wrt y
  def dy_sigmoid_tanh(x,y):
    return sigmoid(x)*(sech(y)**2)
    
Now we can instantiate the module, and use it to compute linear bounds.
  
  # The dReal configuration.
  drCfg = [("use_polytope", True), ("precision", 0.0000001), ("number_of_jobs", 1)]
  
  # All of the None arguments are for parameters that are no longer used. (They were for some features I
  # was experimenting with, but did not end up being useful).
  linsynModule = RuntimeActBounds(sigmoid_tanh, [dx_sigmoid_tanh, dy_sigmoid_tanh], None, 0.1, None, 1e-6, None, drCfg)
  
  # compute a linear bound for sigmoid(x)*tanh(y) when x in [-2, 2] and y in [-2, 2]
  x_bounds = (-2, 2)
  y_bounds = (-2, 2)
  coeffs = linsynModule.computePlanes([x_bounds, y_bounds])
  cl_1, cl_2, cl_3, cu_1, cu_2, cu_3 = coeffs
  print(coeffs)

## auto_lirpa documention
auto_lirpa is an actively developed framework for doing linear bounding-based verification of neural networks. Providing documentation for it is out of the scope of this artifact. If you wish to analyze neural network architectures not considered in our paper, then we recommend reading their documentation: https://auto-lirpa.readthedocs.io/en/latest/.

This artifact contains the January 2021 release of auto_lirpa (there have been several releases between now and when i started this project).
I made some hacks to arm-twist auto_lirpa into allowing me to bound the activation functions as a whole. Specifically, I extend their BoundedModule class to merge some of the computation graph nodes into single nodes.

I do _not_ recommend reusing my modifed auto_lirpa code. The auto_lirpa team just recently (as in three days ago at the time of writing) added a feature to define custom operations that will let you bound activations as a whole, and it is documented here: https://auto-lirpa.readthedocs.io/en/latest/custom_op.html. If you wish to analyze new architectures/activations not considered in this paper I highly recommend learning about auto_lirpa and extending it yourself. 
