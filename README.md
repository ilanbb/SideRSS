# SideRSS
SideRSS: A Siamese deep neural network for predicting RNA structure from high-throughput RNA probing data

A TensorFlow implementation of a Siamese convolutional neural networks for predicting RNA structure probabilities from RNA sequence. 
The network outputs a single value for every nucleotide, indicating its probability of being unpaired. 

Requirements:

	Python, Numpy, Tensorflow (version 1.8).

Setting up:

	Clone the repositopry into your working space.

Training:

	For traininig a prediction model: 
        
        python siamese/main.py train <train_seq_file> <train_struct_path>

	The model will be saved in the current directory under a predefined name.

Evaluation:

	For evaluating the perfomance of a pre-trained model on a test/validation set:

        python siamese/main.py eval <test_seq_file> <test_struct_path>

        A model with a predefined name is taken from the current directory.


Predicting:

        For predicting structure probabilities using a pre-trained model:

        python siamese/main.py predict <test_seq_file> <flank_size> <prediction_file> 

        A model with a predefined name is taken from the current directory. If no flanking was used, set flank value to zero.


Input format:

A sequence-information file contains a set of RNA sequnces. Every sequence is stored in two lines in the following format:

	>1:4687934-4689403
        CCAACTTCATTTTTTATTTGCGCTTGAA...

A structure-information file contains structure probabilities, one for each nucleotide in the corresponding sequences (probability of being unpaired). Information on every sequence is stored in two lines in the following format: 

	>1:4687934-4689403
        0.032	0.02	0.0	0.021	0.0	0.0	0.083...

Partial sample input files can be found in the data directory.

Output format:

The format of a prediction file is similar to the format of a structure information input file, except that it also contains the probability each nucleotide is paired. 
