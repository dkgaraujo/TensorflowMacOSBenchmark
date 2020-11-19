# TensorflowMacOSBenchmark
 Codes used to benchmark performance of Apple's fork TensorFlow 2.4 for MacOS, using an Intel silicon MacBook Pro

## Results

Note: I restarted the computer to try to run the benchmark as clean as possible. However, the results below were run only once, which might theoretically lead some memory management issues to leak from one test into the subsequent one. (In other words, I think ideally a robust benchmark should include multiple runs, where the order of the benchmark is randomly selected each time.) But for practical purposes, the differences in performance can already be informative.

`Training a simple CNN model with gpu: 237.959 sec elapsed`
`Training a simple CNN model with cpu: 211.433 sec elapsed`
`Training a simple CNN model with plaidml using the GPU: 483.239 sec elapsed`

The test was conducted on Thursday, 19 November 2020.

## Code to replicate the benchmark

There are two files. `CNN.R` implements the [tensorflow 2.4 + MacOS](https://github.com/apple/tensorflow_macos) add-on with CPU and GPU (you can choose at the beginning of the code), whereas `CNN_plaidml.R` implements the [PlaidML](https://github.com/plaidml/plaidml) version.

Feel free to fork, send pull requests etc. If you run this example on M1 machines, please consider sharing the results as well, especially if you take care to restart your machine before running the code.

## Gear

CPU: 2.4 GHz 8-Core Intel Core i9
GPU: AMD Radeon Pro 5500M

## Other information

* Note that this code is a convolutional neural network. Other architectures might have different results. In addition, it would be nice to have a comparison amongst tensorflow implementations of probabilistic programming (for example, running a Hamiltonian Monte Carlo algorithm on the CPU and on the GPU.)
* Further, note also that both tensorflow and PlaidML used only the AMD Radeon GPU. If someone knows how to also use the Intel GPU chip (or both together), I would be happy to test that as well for comparison.
* On my *subjective* evaluation, the tensorflow with GPU was the least battery-intensive amongst the three in this exercise, but I did not explicitly document battery usage.
