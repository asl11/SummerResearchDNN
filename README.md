# SummerResearchDNN
Code from the decoder and decoder optimization methods used to predict mouse behavioral state from neuron firing rate. Code is edited code from Kording lab github, with instruction/data processing from Hanlin Zhu.

Decoder.py outlines many neural network methods, we used & edited the DNN_Hanlin method. This file sets up a Keras model, adds layers & other hyperparameters.
TemfilDNN uses the BayesianOptimization package to optimize the hyperparameters in Decoder.py, with manual tweaking by me. Full details in powerpoint presentation at _

Full Credits for the project:

Batty, Eleanor, et al. "BehaveNet: nonlinear embedding and Bayesian neural decoding of behavioral videos." Advances in Neural Information Processing Systems. 2019.
Glaser, Joshua I., et al. "Machine learning for neural decoding." arXiv preprint arXiv:1708.00909 (2017).
Guo, Xifeng, et al. "Deep clustering with convolutional autoencoders." International conference on neural information processing. Springer, Cham, 2017.
He, Fei, et al. "Ultraflexible neural electrodes for long-lasting intracortical recording." iScience (2020): 101387.
Richards, Blake A., et al. "A deep learning framework for neuroscience." Nature neuroscience 22.11 (2019): 1761-1770.
