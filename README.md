# CS273B

Technology for engineering de novo DNA sequences has rapidly advanced in the past several decades, raising the possibility of creating sequences that provide new or improved biological functions.

-----
## GAN
Filename: GAN.ipynb

A generative adversarial network (GAN) consists of a generator and a discriminator. The core concept of a GAN is that the data (in this case, DNA sequences) can be encoded into a real-valued, latent space. We designed our GAN to address the issues with using unsupervised learning to generate DNA sequences: as explained above, the unstructured nature of DNA (almost any possible sequence of nucleotides defines a valid sequence) makes discriminating synthetic examples from real examples challenging. Our generator pushes a randomly initialized vector through four convolutional upsampling layers to create a 1000 base pair long generated sequences. The sequences are then fed into a discriminator that uses three convolutional layers with 0.3 dropout to create a single output classifying the validity of the sequence. The model is trained using an Adam optimizer with a learning rate of 0.0001. The model currently runs for 20 epochs with a 256 batch size. 

## Genetic Algorithms
Filename: genalg.ipynb

Filename: exhaustive.py

To initialize our population, we randomly generated 100 DNA sequences of 1000 nucleotides. In each generation, we mutated each member of the population at a random locus. We then evaluated cell-type specific activity using the provided Basenji model and determined sequence fitness using one of two objective function outlined in our paper. For each sequence, we kept the mutation if it improved sequence fitness and reverted to the sequence from the previous generation if the mutation did not improve fitness. If the algorithm is set to run Metropolis sampling, then it will accept as mentioned previously and reject with a probability proporational to the exponentiated changed in objective function. If the algorithm is set to culling, it will cull all samples that did not improve fitness and replace with some probability. We ran either 100 or 1000 generations and maintained a population size of 100.

## Backpropagation
Filename: input_backprop.ipynb

We applied backpropagation to the Basenji model (https://github.com/calico/basenji/tree/tf2) provided by Dr. David Kelley and to our own implementation of the Basenji architecture using the same hyperparameters as Dr. Kelley. To introduce stochasticity into the sequence generation, we performed backpropagation 50 times with a different random normal initialization of the input sequence, where each input sequence is a 1000-by-4 matrix. During each iteration, we maintained two sequences: a real-valued sequence and a one-hot encoded sequence. To generate an initial one-hot encoded sequence from the initial real-valued sequence, at each position we selected the base corresponding to the real-valued element with the maximum value. Each iteration, we calculated the loss with respect to the one-hot encoded sequence but updated the real-valued sequence. Every 100 iterations, we checked the real-valued sequence to determine if the location of the maximum value in each row changed and updated the one-hot encoded sequence accordingly. During backpropagation, we used an Adam optimizer with a step-size of $1e-4$. We updated for a minimum of 40000 iterations or until the loss did not decrease by 0.1\% between iterations.

