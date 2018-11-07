import numpy as np

class Dataset():
    """
    Class for managing training/testing batches
    """

    def __init__(self, sequences, comp_sequences, lengths, structures, batch_size, batch_limit, env_size):
        '''
        CTOR receiving and soring all relveant data
        Args:
            sequences: Numpy array of sequence matrices (Python nested list), each of shape [sequence_length, alphabet_size].
            comp_sequences: Numpy array of complement sequence matrices (Python nested list), each of shape [sequence_length, alphabet_size].
            lengths: Numpy array of sequence lengths.
            structures: Numpy array of structure matrices each of shape [sequence_length , structures_num].
            batch_size: maximum length of a batch
            batch_limit: maximum total number of nucleotides in a batch
            env_size: radius of environment that is analuzed for every nucleotide
        '''
        self.sequences = sequences
        self.comp_sequences = comp_sequences
        self.lengths = lengths
        self.structures = structures
        self.batch_size = batch_size
        self.limit = batch_limit
        self.data_size = sequences.shape[0]
        self.K = env_size
        self.curr_ind = 0


    def has_next_batch(self):
        '''
        Checks whether there are more data left to provide        
        '''
        return self.curr_ind != self.data_size


    def reset(self):
        '''
        Reset state in order to provide data batches from scratch
        '''
        self.curr_ind = 0


    def get_pad_sizes(self, max_len, cur_len):
        '''
        Compute padding size for a specific sequence according to the maximum sequence length.

        Args:
            max_len: maximum length of a sequence (can be batch specific)
            cur_len: length of padded sequence

        Returns:
            left: length of left padding.
            right: length of right padding.
        '''
        diff = max_len - cur_len
        left, right = 0, 0
        if diff > 0:
            if diff % 2 == 0:
                left, right = diff/2, diff/2
            else:
                left, right = diff/2+1, diff/2
        return left, right

    
    def pad_samples(self, samples, max_sample_len):
        '''
        Process a batch of squences according to the maximum sequence length and radis of analysis environment.

        Args:
            samples: Numpy array of sequence matrices (represented as Python nested lists).
            max_sample_len: maximum length of a sequence.

        Returns:
            padded_samples: 3D numpy array of padded samples of shape [samples_num, (2K+max_sequence_length), alphabet_size].
            masks: 2D numpy array of masks of shape [samples_num, max_sequence_length] where a False value signals a padded position.
        '''
        padded_samples = list()
        masks = list()
        for sample in samples:
            cur_sample_len = len(sample)
            # Pad sample so that it would be of length max_sample_len
            left, right = self.get_pad_sizes(max_sample_len, cur_sample_len)
            temp_sample = np.concatenate([np.concatenate([np.zeros((left, 4))+0.25, np.array(sample)], axis=0), np.zeros((right, 4))+0.25], axis=0)
            # Pad sample to allow K-environment analysis on both sides
            temp_sample = np.concatenate([np.concatenate([np.zeros((self.K, 4))+0.25, temp_sample], axis=0), np.zeros((self.K, 4))+0.25], axis=0)
            padded_samples.append(temp_sample)
            # Create the padding mask (of length max_sample_len)
            mask = [False]*left + [True]*cur_sample_len + [False]*right
            masks.append(mask)
        padded_samples = np.array(padded_samples)
        masks = np.array(masks)
        return padded_samples, masks
    

    def next_batch(self):
        '''
        Provide next batch of data
        
        Returns:
            seqs_batch: batch of sequences of shape [batch_size, max_seq_len + 2K, 4]
            lengths_batch: batch of sequence lengths of shape [batch_size]
            structs_batch: batch of structure information for each nucletode (only unpaired probability) of shape [total_num_nucleotides]
            masks_batch: batch of padding data (masks) of shape [batch_size, max_seq_len]
            rev_seqs_batch: batch of reverse comlement sequences of shape [batch_size, max_seq_len + 2K, 4]
            curr_batch_size: size of current batch
            curr_max_len: maximum length of a sequence in current batch
        '''
        end_ind = self.curr_ind
        total_bases = 0
        # Next batch should: 
        # (1) be not more than batch size
        # (2) no overflow 
        # (3) not exceed the limit on total number of nucleotides
        while end_ind < self.data_size and total_bases < self.limit and end_ind - self.curr_ind < self.batch_size:
            total_bases += self.lengths[end_ind]
            end_ind += 1

        seqs_batch = self.sequences[self.curr_ind:end_ind]
        comp_seqs_batch = self.comp_sequences[self.curr_ind:end_ind]
        lengths_batch = self.lengths[self.curr_ind:end_ind]
        structs_batch = self.structures[self.curr_ind:end_ind]

        # Compute batch statistics
        curr_batch_size = len(seqs_batch)
        curr_max_len = max(lengths_batch)

        # Pad every sample according to batch statistics to make sequence batch of shape [batch_size, max_seq_len + 2K, 4]
        seqs_batch, masks_batch = self.pad_samples(seqs_batch, curr_max_len)
        comp_seqs_batch, _ = self.pad_samples(comp_seqs_batch, curr_max_len)
        # Reverse the complement sequences
        rev_seqs_batch = np.flip(comp_seqs_batch, 1)

        # Concatenate all structure probabilties to a vector of shpe [total_num_nucleotides]
        # Note: this code is currently tailored to prediction of only one probability per nucleotide
        structs_batch = np.squeeze(np.concatenate(structs_batch, axis=0), axis=1)

        # Prepeare to next batch analysis
        self.curr_ind = end_ind

        return seqs_batch, lengths_batch, structs_batch, masks_batch, rev_seqs_batch, curr_batch_size, curr_max_len
