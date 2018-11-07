import random
import numpy as np
import tensorflow as tf


MIN_SEQ_LEN = 10
MAX_SEQ_LEN = 30000


def argsort(seq):
    '''
    Argsort implementation for a Python list.

    Args:
        seq: Python list.

    Returns:
        list of indices of sorted elements.
    '''
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]


def transpose(mat):
    '''
    Transpose Python list.

    Args:
        mat: 2D matrix represented as a Python list of lists.

    Returns:
        trans: Transposed 2D matrix also represented as a Python list of lists.
    '''
    num_mat = np.array(mat)
    num_trans = num_mat.transpose()
    trans = [list(row) for row in num_trans]
    return trans



def process_transcriptome_header(header):
    '''
    Process transcriptome header line.

    Args:
        header: header string of the form >CHOROMOSOM:START_INDEX-END_INDEX.

    Returns:
        tuple containing the chromosome number, start index and end index.
    '''
    assert(header[0] == '>')
    chromosome_end_index = header.find(":")
    indices_sep_index = header.find("-")
    chromosome = header[1:chromosome_end_index]
    start_index = header[chromosome_end_index+1:indices_sep_index]
    end_index = header[indices_sep_index+1:]
    return chromosome, start_index, end_index


def process_annotations(annotations_file, structures_num):
    '''
    Process next transcriptome and extract meta data and structure probabilities.

    Args:
        annotations_file: handler to a structure probabilities file.
            Each transcriptome data is stored in the following format:
            (-) Header line.
            (-) Structure probabilities line for each of the possible structure contexts
        structures_num: number of possible structure contexts, e.g., paired and unpaired.

    Returns:
        transcriptome_data: tuple with transcriptome chromosome, start endex and end index.
        structure_matrix: structure probabilities matrix of shape sequence_length X structures_num.
            The matrix is represented as a Python nested list.
        if no data is left to be read from the file, None value is returned.
    '''
    data = annotations_file.readline()
    if not data:
        return None
    header_line = data.strip()
    transcriptome_data = process_transcriptome_header(header_line)
    matrix = list()
    for structure_index in range(structures_num):
        structure_line = annotations_file.readline().strip()
        matrix_line = [float(elem) for elem in structure_line.split()]
        matrix.append(matrix_line)
    structure_matrix = transpose(matrix)
    return (transcriptome_data, structure_matrix)


def process_sequences(sequences_file):
    '''
    Process next transcriptome and extract meta data and sequence data.

    Args:
        sequences_file: handler to a sequence file
            Each transcriptome data is stored in the following format:
            (-) Header line.
            (-) Sequence line.

    Returns:
        transcriptome_data: tuple with transcriptome chromosome, start endex and end index.
        seq_matrix: sequence matrix of shape sequence_length X alphabet_size.
            The matrix is represented as a Python nested list.
        comp_seq_matrix: complement sequence matrix of shape sequence_length X alphabet_size.
            The matrix is represented as a Python nested list.
        if no data is left to be read from the file, None value is returned.
    '''
    data = sequences_file.readline()
    if not data:
        return None
    header_line = data.strip()
    transcriptome_data = process_transcriptome_header(header_line)
    seq = sequences_file.readline().strip()
    seq_matrix = list()
    comp_seq_matrix = list()
    for base in seq:
        base = base.upper()
        if base == 'A':
            base_encoding = [1, 0, 0, 0]
            comp_base_encoding = [0, 0, 0, 1]
        elif base == 'C':
            base_encoding = [0, 1, 0, 0]
            comp_base_encoding = [0, 0, 1, 0]
        elif base == 'G':
            base_encoding = [0, 0, 1, 0]
            comp_base_encoding = [0, 1, 0, 0]
        elif base == 'U' or base == 'T':
            base_encoding = [0, 0, 0, 1]
            comp_base_encoding = [1, 0, 0, 0]
        elif base == 'N':
            base_encoding = [1, 1, 1, 1]
            comp_base_encoding = [1, 1, 1, 1]
        else:
            raise ValueError("Base is " + base)
        seq_matrix.append(base_encoding)
        comp_seq_matrix.append(comp_base_encoding)
    return (transcriptome_data, seq_matrix, comp_seq_matrix)


def read_data(sequences_file, annotations_file, structures_num, sorting=True):
    '''
    Read icSHAPE data for training (sequence and structure information)

    Args:
        sequences_file: file containing transcriptome sequence information.
        annotations_file: file containing transcriptome structure information.
        structures_num: number of possible structure contexts, e.g., paired and unpaired.
        sorting: whether to sort the data in an increasing order of sequence length.

    Returns:
        sequences: Numpy array of sequence matrices (Python nested list), each of shape sequence_length X alphabet_size.
        comp_sequences: Numpy array of complement sequence matrices (Python nested list), each of shape sequence_length X alphabet_size.
        lengths: Numpy arrayof sequence lengths.
        structures: Numpy array of structure matrices each of shape sequence_length X structures_num.
    '''
    with open(sequences_file) as seq_data, open(annotations_file) as annot_data:
        sequences, comp_sequences, lengths, structures = list(), list(), list(), list()
        while True:
            # Process next sample information:  sequence and structure annotations.
            data_seq = process_sequences(seq_data)
            data_struct = process_annotations(annot_data, structures_num)
            # No more samples to process - return entire data
            if not data_seq or not data_struct:
                sequences, comp_sequences, lengths, structures = np.array(sequences), np.array(comp_sequences), np.array(lengths), np.array(structures)
                # Sort by sequence length if necessary
                if sorting:
                    indices = argsort(lengths)
                    lengths = lengths[indices]
                    sequences = sequences[indices]
                    comp_sequences = comp_sequences[indices]
                    structures = structures[indices]
                return sequences, comp_sequences, lengths, structures
            transcriptome_data_1, seq_matrix, comp_seq_matrix = data_seq
            transcriptome_data_2, struct_matrix = data_struct
            # Validate transcriptome identity
            assert(transcriptome_data_1 == transcriptome_data_2)
            # Compute sequnce length
            curr_seq_len = len(seq_matrix)
            # Skip too short and too long transcriptomes
            if curr_seq_len < MIN_SEQ_LEN or curr_seq_len > MAX_SEQ_LEN:
                continue
            # Aggregate new transcriptome data 
            lengths.append(curr_seq_len)
            sequences.append(seq_matrix)
            comp_sequences.append(comp_seq_matrix)
            structures.append(struct_matrix)
        assert(False)


def read_data_seq_only(sequences_file, sorting=True):
    '''
    Read icSHAPE data for testing (sequence information only)

    Args:
        sequences_file: file containing transcriptome sequence information.
        sorting: whether to sort the data in an increasing order of sequence length.

    Returns:
        sequences: Numpy array of sequence matrices (python nested list), each of shape sequence_length X alphabet_size.
        comp_sequences: Numpy array of complement sequence matrices (Python nested list), each of shape sequence_length X alphabet_size.
        lengths: Numpy array of sequence lengths.
        structures: Numpy array of dummy structure matrices each of shape sequence_length X structures_num (filled with zeros).
    '''

    with open(sequences_file) as seq_data:
        sequences, comp_sequences, lengths, structures = list(), list(), list(), list()
        while True:
            # Process next sample: sequence information only
            data_seq = process_sequences(seq_data)
            # No more samples to process - return entire data
            if not data_seq:
                sequences, comp_sequences, lengths, structures = np.array(sequences), np.array(comp_sequences), np.array(lengths), np.array(structures)
                # Sort by sequence length if necessary
                if sorting:
                    indices = argsort(lengths)
                    lengths = lengths[indices]
                    sequences = sequences[indices]
                    comp_sequences = comp_sequences[indices]
                    structures = structures[indices]
                return sequences, comp_sequences, lengths, structures
            transcriptome_data_1, seq_matrix, comp_seq_matrix = data_seq
            # Compute sequnce length
            curr_seq_len = len(seq_matrix)
            # Skip too short and too long transcriptomes
            if curr_seq_len < MIN_SEQ_LEN or curr_seq_len > MAX_SEQ_LEN:
                continue
            # Aggregate new transcriptome data (with dummy structure matrix of zeros).
            lengths.append(curr_seq_len)
            sequences.append(seq_matrix)
            comp_sequences.append(comp_seq_matrix)
            structures.append([[0.0] for i in range(curr_seq_len)])
        assert(False)
