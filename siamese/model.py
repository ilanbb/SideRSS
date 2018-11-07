import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core
from dataset import Dataset
from utils import *
from cnn_utils import *
import math
import os
import scipy.stats

ALL_DATA = 10000000

class SideRSS:

    '''
    SideRSS: RNA secondary structure from sequence using Siamese convolutional neural network
    '''

    def __init__(self, train_seq_path, test_seq_path, train_struct_path, test_struct_path, params, network_name, conv_layers, strides_x_y):
        '''
        SideRSS CTOR

        Args:
            train_seq_path: path of a sequence information file for training
            test_seq_path: path of a sequence information file for testing
            train_struct_path: path of a structure information file for training
            test_struct_path: path of a structure information file for testing
            params: dictionary of network parameters
            network_name: name of network, used for saving the network
            conv_layers: array of dictionaries with information on each convolutional layer
            strides_x_y: tuple of (x, y) strides used in each of the conolutional layers
        '''
        self.train_seq_path = train_seq_path
        self.test_seq_path = test_seq_path
        self.train_struct_path = train_struct_path
        self.test_struct_path = test_struct_path
        self.params = params
        self.network_name = network_name
        self.conv_layers = conv_layers
        self.strides_x_y = strides_x_y
        # Initialize variables of all convolutional layers
        self.conv_weights = list()
        self.conv_biases = list()
        for layer_index in range(len(self.conv_layers)):
            weights, biases = self.init_conv_layer(layer_index)
            self.conv_weights.append(weights)
            self.conv_biases.append(biases)
        # Initialize variables of hidden layers
        self.hidden_weights = list()
        self.hidden_biases = list()
        hidden_data = list()
        hidden_data.append(self.init_hidden_layer(self.params['top_k'], self.params['hidden_layers_sizes'][0]))
        hidden_data.append(self.init_hidden_layer(self.conv_layers[1]['output_channels'], self.params['hidden_layers_sizes'][1]))
        for layer_index in range(len(self.params['hidden_layers_sizes'])):
            self.hidden_weights.append(hidden_data[layer_index][0])
            self.hidden_biases.append(hidden_data[layer_index][1])
        # Initialize variables for final output layer
        self.final_output_weight, self.final_output_bias = self.init_hidden_layer(self.params['hidden_layers_sizes'][1], 1)
        # Initialize matrix for siamese channels merge
        self.merge_matrix = tf.Variable(tf.truncated_normal([self.conv_layers[0]['output_channels'], self.conv_layers[0]['output_channels']], stddev=0.1))
        self.merge_matrix = tf.identity(self.merge_matrix, name="mergematrix")

       
    def init_conv_layer(self, conv_layer_index):
        '''
        Initiailize parameters of a convolutional layer: filter weights and bias

        Args:
            conv_layer_index: index of convolutional layer in the network

        Returns:
            Tuple of convolutional layer parameters: weights and biases
        '''
        weights = list()
        biases = list()
        # Extract layer parameters
        input_channels = self.conv_layers[conv_layer_index]['input_channels']
        output_channels = self.conv_layers[conv_layer_index]['output_channels']
        filter_x = self.conv_layers[conv_layer_index]['size_x']
        filter_y = self.conv_layers[conv_layer_index]['size_y']
        #weights = tf.identity(tf.Variable(tf.truncated_normal([filter_x, filter_y, input_channels, output_channels], stddev=0.1)), name="WEIGHTS" + str(conv_layer_index))
        limit = math.sqrt(6.0 / (filter_x*filter_y*input_channels + filter_x*filter_y*output_channels))
        weights = tf.Variable(tf.random_uniform(shape=[filter_x, filter_y, input_channels, output_channels], minval=-limit, maxval=limit))
        weights = tf.identity(weights, name="conv_weights" + str(conv_layer_index))
        biases = tf.identity(tf.Variable(tf.zeros(output_channels)), name="conv_biases" + str(conv_layer_index))
        return weights, biases


    def init_hidden_layer(self, weight_x, weight_y):
        '''
        Initiailize parameters of a hidden layer: weights and bias

        Args::
            weight_x: number of rows in the the weight matrix 
            weight_y: number of columns in the the weight matrix 

        Returns:
            Tuple of layer parameters: weights and biases
        '''
        weights = tf.Variable(tf.truncated_normal([weight_x, weight_y], stddev=0.1))
        biases = tf.Variable(tf.zeros(1))
        return weights, biases


    def train(self):
        '''
         Train Siamese convolutional network to predict RNA secondary structure from sequence
         Model is stored in recovery files after training
        '''
        # Get train data
        train_seqs, train_comp_seqs, train_lengths, train_structs = read_data(self.train_seq_path, self.train_struct_path, self.params['rna_structures']) 

        # Create dataset object for batching
        dataset = Dataset(train_seqs, train_comp_seqs, train_lengths, train_structs, self.params['batch_size'], self.params['batch_limit'], self.params['conv_env_sizes'][0])

        # Define placeholders for data, lengths and structure probability vectors
        # Sequences and their reverse complement should be a tensor of shape [batch_size, batch_max_seq_len + 2*conv_env, rna_alphabet_size]
        # Lengths should be a tensor of shape [batch_size]
        # Structures should be a tensor of shape [total_num_nucleotides]
        # Masks should be a tensor of shape [batch_size, batch_max_seq_len]
        self.rna_sequences = tf.placeholder(tf.float32, [None, None, self.params['rna_bases']], name="rnaseqs")
        self.rna_rc_sequences = tf.placeholder(tf.float32, [None, None, self.params['rna_bases']], name="rnarcseqs")
        self.rna_lengths = tf.placeholder(tf.int32, [None], name="rnalengths")
        self.rna_structures = tf.placeholder(tf.float32, [None], name="rnastructs")
        self.rna_masks = tf.placeholder(tf.bool, [None, None], name="rnamasks")

        # Compute actual batch size for current batch
        actual_batch_size_int = tf.to_int32(tf.shape(self.rna_lengths)[0])

        # Convert sequence data tensors from 3D into 4D (for convolution with 1 input channel)
        self.rna_sequences_data = tf.expand_dims(self.rna_sequences, -1)
        self.rna_rc_sequences_data = tf.expand_dims(self.rna_rc_sequences, -1)

        # Run first copy of Siamese CNN network on sequence data and obtain output of shape [batch_size, batch_max_seq_len, 1, num_output_channels]
        self.conv_output = tf.identity(create_conv_layer(self.rna_sequences_data, self.conv_weights[0], self.conv_biases[0], self.strides_x_y), name="rnaconv1")

        # Squeeze convolutional output to shape [batch_size, batch_max_seq_len, num_output_channels]
        self.conv_output = tf.squeeze(self.conv_output, axis=2, name="rnaconv1squeezed")

        # Run second copy of Siamese CNN network on reverse comlement sequence data and obtain output of shape [batch_size, batch_max_seq_len, num_output_channels]
        self.conv_rc_output = tf.identity(create_conv_layer(self.rna_rc_sequences_data, self.conv_weights[0], self.conv_biases[0], self.strides_x_y), name="rnarevconv1")
        self.conv_rc_output = tf.squeeze(self.conv_rc_output, axis=2, name="rnaconvrc1squeezed")

        # Expand merge matrix to multiply each sample in the batch
        self.expanded_merge_matrix = tf.tile(tf.expand_dims(self.merge_matrix, 0), multiples=[actual_batch_size_int, 1, 1])

        # Merge information from two copies of the Siamese networks by multiplying sequence matrix, merge matrix and transpose of reverse complement sequence matrix
        self.combined_data = tf.matmul(tf.matmul(self.conv_output, self.expanded_merge_matrix), tf.transpose(self.conv_rc_output, perm=[0,2,1]))

        # Max-k pooling to reduce shape from [batch_size, batch_max_len, batch_max_len] to [batch_size, batch_max_len, top_k]
        self.combined_data, _ = tf.nn.top_k(self.combined_data, k=self.params['top_k'])

        # Apply fully connected layer on each nucleotidec: [batch_size, batch_max_len, top_k] X [batch_size, top_k, hidden_size] = [batch_size, batch_max_len, hidden_size]
        self.hidden_layer_1 = nn_layer(self.combined_data, tf.tile(tf.expand_dims(self.hidden_weights[0], 0), multiples=[actual_batch_size_int, 1, 1]), self.hidden_biases[0], True)
        self.hidden_layer_1 = tf.identity(self.hidden_layer_1, name="rnafc1")

        # Expand and pad matrix for narrow convolution that would yield output of the same length as the number of nucleotides 
        pad_matrix = tf.zeros([actual_batch_size_int, self.params['conv_env_sizes'][1], self.params['hidden_layers_sizes'][0], 1], dtype=tf.float32)
        self.conv_2_input = tf.concat([tf.concat([pad_matrix, tf.expand_dims(self.hidden_layer_1, 3)], axis=1), pad_matrix], axis=1)

        # Apply second convolutional layer and get output of shape [batch_size, batch_max_len, num_output_channels]
        self.conv_2_output = create_conv_layer(self.conv_2_input, self.conv_weights[1], self.conv_biases[1], self.strides_x_y)
        self.conv_2_output = tf.squeeze(self.conv_2_output, axis=2)

        # Extract [total_num_nucleotides, first_hidden_size] from the output of the second convolutional layer
        self.masked_output = tf.boolean_mask(self.conv_2_output, self.rna_masks)

        # Apply a second fully connected layer on each nucleotide
        self.hidden_layer_2 = tf.identity(nn_layer(self.masked_output, self.hidden_weights[1], self.hidden_biases[1], True), name="rnafc2")

        # Output predictions as a tensor of shape [total_num_nucleotides]
        self.predictions = tf.identity(nn_layer(self.hidden_layer_2, self.final_output_weight, self.final_output_bias, False), name="rnafc3")
        self.predictions = tf.identity(tf.squeeze(self.predictions, axis=1), name="rnapredictions")

        # Flatten structure probabilities
        self.flat_rna_structures = tf.reshape(self.rna_structures, [-1], name="rnaflatstructs")

        # Compute loss
        regularization = tf.nn.l2_loss(self.hidden_weights[0]) + tf.nn.l2_loss(self.hidden_weights[1])
        loss = tf.identity(tf.reduce_mean(tf.square(self.predictions - self.flat_rna_structures) + self.params['beta']*regularization), name="rnaloss")

        # Compute accuracy
        accuracy = tf.identity(tf.reduce_mean(tf.abs(self.predictions - self.flat_rna_structures)), name="rnaaccuracy")

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.params['max_gradient_norm'])

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        train_step = optimizer.apply_gradients(zip(clipped_gradients, params))

        #init = tf.global_variables_initializer()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Create a saver object to store optimal traininig network
        saver = tf.train.Saver()

        # Create a new session with the configuration graph
        with tf.Session() as sess:
                    
            # Start trainig by initializing the session
            sess.run(init)

            # Perform epochs
            for epoch_index in range(self.params['num_epochs']):
               	batch_counter = 0
                while dataset.has_next_batch():
                            
                    batch_counter += 1

                    # Read next batch
                    seqs_batch, lengths_batch, structs_batch, masks_batch, rc_seqs_batch, curr_batch_size, curr_max_len = dataset.next_batch()
                    feed_dict = {self.rna_sequences: seqs_batch, self.rna_rc_sequences: rc_seqs_batch, self.rna_lengths: lengths_batch, self.rna_structures: structs_batch, self.rna_masks: masks_batch}

                    # Perform batch optimization
                    sess.run(train_step, feed_dict=feed_dict)

                    # Output stats every once in a while
                    if batch_counter % self.params['stop_check_interval'] == 0:

                        # Compute batch loss and accuracy
                        batch_loss, batch_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict)

                        # statistics
                        status = 'Train acc: {}\tTrain Loss: {}\t{}\t{}\t{}'.format(batch_accuracy, batch_loss, batch_counter, dataset.curr_ind, epoch_index)
                        print (status)

                # Reset dataset for a possible next epoch            
                dataset.reset()

            # Save model
            save_path = saver.save(sess, self.network_name)


    '''
    Evaluate performance of current model on a given dataset (validation/testing)

    Returns:
        tuple with results on tests/validation set: loss and accuracy
    '''
    def test(self):

        # Read all test data in one batch
        test_seqs, test_comp_seqs, test_lengths, test_structs = read_data(self.test_seq_path, self.test_struct_path, self.params['rna_structures'], sorting=False) 
        dataset = Dataset(test_seqs, test_comp_seqs, test_lengths, test_structs, ALL_DATA, ALL_DATA, self.params['conv_env_sizes'][0])
        seqs_batch, lengths_batch, structs_batch, masks_batch, rc_seqs_batch, curr_batch_size, curr_max_len = dataset.next_batch()

        # Create a new session 
        tf.reset_default_graph()
        new_graph = tf.Graph()
        with tf.Session(graph=new_graph) as eval_sess:

            # Import the trained network
            network = tf.train.import_meta_graph(self.network_name + '.meta')

            # Load the trained parameters
            network.restore(eval_sess, self.network_name)

            # Construct feed data using placeholder names
            rna_sequences = new_graph.get_tensor_by_name("rnaseqs:0")
            rna_rc_sequences = new_graph.get_tensor_by_name("rnarcseqs:0")
            rna_lengths = new_graph.get_tensor_by_name("rnalengths:0")
            rna_structures = new_graph.get_tensor_by_name("rnastructs:0")
            rna_masks = new_graph.get_tensor_by_name("rnamasks:0")
            feed_dict={rna_sequences: seqs_batch, rna_rc_sequences: rc_seqs_batch, rna_lengths: lengths_batch, rna_structures: structs_batch, rna_masks:masks_batch}

            # Access the evaluation metric
            test_loss = new_graph.get_tensor_by_name("rnaloss:0")
            test_accuracy = new_graph.get_tensor_by_name("rnaaccuracy:0")
            loss, accuracy = eval_sess.run([test_loss, test_accuracy], feed_dict)

            return loss, accuracy


    '''
    Evaluation in batches of test/validation set

    Returns:
        tuple with results on test/validation seit:
            mse: mean squared error
            mae: mean absolute error
            pr:  Person correlation
    '''
    def batch_eval(self):

        # Set higher values to batch size and limit such that no OOM Exception happens
        self.params['batch_size'] = 5 * self.params['batch_size']
        self.params['batch_limit'] = 5 * self.params['batch_limit']

        # Read test data
        test_seqs, test_comp_seqs, test_lengths, test_structs = read_data(self.test_seq_path, self.test_struct_path, self.params['rna_structures'], sorting=False)
        dataset = Dataset(test_seqs, test_comp_seqs, test_lengths, test_structs, self.params['batch_size'], self.params['batch_limit'], self.params['conv_env_sizes'][0])

        # Create a new session 
        tf.reset_default_graph()
        new_graph = tf.Graph()

        # Initialize data accumalators
        batch_counter = 0
        total_seqs = 0
        total_bases = 0
        total_diff = 0
        total_squared_diff = 0
        all_structs = np.array([])
        all_preds = np.array([])
 
        # Init test session
        with tf.Session(graph=new_graph) as eval_sess:

            while dataset.has_next_batch():
                batch_counter += 1

                seqs_batch, lengths_batch, structs_batch, masks_batch, rc_seqs_batch, curr_batch_size, curr_max_len = dataset.next_batch()

                # Import the trained network
                network = tf.train.import_meta_graph(self.network_name + '.meta')

                # Load the trained parameters
                network.restore(eval_sess, self.network_name)

                # Construct feed data using placeholder names
                rna_sequences = new_graph.get_tensor_by_name("rnaseqs:0")
                rna_rc_sequences = new_graph.get_tensor_by_name("rnarcseqs:0")
                rna_lengths = new_graph.get_tensor_by_name("rnalengths:0")
                rna_structures = new_graph.get_tensor_by_name("rnastructs:0")
                rna_masks = new_graph.get_tensor_by_name("rnamasks:0")
                con_mat = new_graph.get_tensor_by_name("mergematrix:0")
                feed_dict={rna_sequences: seqs_batch, rna_rc_sequences: rc_seqs_batch, rna_lengths: lengths_batch, rna_structures: structs_batch, rna_masks:masks_batch}

                # Access the predicetion results
                predictions_node = new_graph.get_tensor_by_name("rnapredictions:0")
                lengths_node = new_graph.get_tensor_by_name("rnalengths:0")
                predictions, lengths, merge_matrix = eval_sess.run([predictions_node, lengths_node, con_mat], feed_dict)

                # Update data accumalators with current batch statistics
                total_diff += np.sum(np.absolute(structs_batch - predictions))
                total_squared_diff += np.sum(np.square(structs_batch - predictions))
                total_seqs += len(lengths)
                total_bases += predictions.shape[0]
                all_structs = np.concatenate((all_structs, structs_batch))
                all_preds = np.concatenate((all_preds, predictions))

                # Report progress
                print "Done with", total_seqs, "sequences and", total_bases, "nucleotides"

            # Compyte statistics on entire test/validation set
            mse = 1.0 * total_squared_diff / total_bases
            mae = 1.0 * total_diff / total_bases
            pr = scipy.stats.pearsonr(all_structs, all_preds)

            # Store merge matrix
            np.save("merge_matrix", merge_matrix)

            return mse, mae, pr

    '''
    Predict structure probabilities given a model and sequence data, and store them into a file

    Args:
        prediction_file: the file to store predictions at
        flank: integer indicating how many bases flank the original RNA sequences from each side
        struct_num: numbr of structure contexts
    '''
    def predict(self, prediction_file, flank, struct_num=2):

        # Remove previous predictions if exists
        try:
            os.remove(prediction_file)
        except:
            pass

        # Set higher values to batch size and limit such that no OOM Exception happens
        self.params['batch_size'] = 5 * self.params['batch_size']
        self.params['batch_limit'] = 5 * self.params['batch_limit']

        # Read test data
        real_seqs, real_comp_seqs, real_lengths, dummy_structs = read_data_seq_only(self.test_seq_path, sorting=False)
        dataset = Dataset(real_seqs, real_comp_seqs, real_lengths, dummy_structs, self.params['batch_size'], self.params['batch_limit'], self.params['conv_env_sizes'][0])

        # Create a new session 
        tf.reset_default_graph()
        new_graph = tf.Graph()

        # Initialize data accumalators
        batch_counter = 0
        total_bases = 0
        total_seqs = 0
        with tf.Session(graph=new_graph) as eval_sess:

            while dataset.has_next_batch():
                batch_counter += 1

                seqs_batch, lengths_batch, structs_batch, masks_batch, rc_seqs_batch, curr_batch_size, curr_max_len = dataset.next_batch()

                # Import the trained network
                network = tf.train.import_meta_graph(self.network_name + '.meta')

                # Load the trained parameters
                network.restore(eval_sess, self.network_name)

                # Construct feed data using placeholder names
                rna_sequences = new_graph.get_tensor_by_name("rnaseqs:0")
                rna_rc_sequences = new_graph.get_tensor_by_name("rnarcseqs:0")
                rna_lengths = new_graph.get_tensor_by_name("rnalengths:0")
                rna_structures = new_graph.get_tensor_by_name("rnastructs:0")
                rna_masks = new_graph.get_tensor_by_name("rnamasks:0")
                feed_dict={rna_sequences: seqs_batch, rna_rc_sequences: rc_seqs_batch, rna_lengths: lengths_batch, rna_structures: structs_batch, rna_masks:masks_batch}

                # Access the predicediction results
                predictions_node = new_graph.get_tensor_by_name("rnapredictions:0")
                lengths_node = new_graph.get_tensor_by_name("rnalengths:0")
                predictions, lengths = eval_sess.run([predictions_node, lengths_node], feed_dict)

                # Accumalate data over batches 
                total_seqs += len(lengths)
                total_bases += predictions.shape[0]

                # Report progress
                print "Done with", total_seqs, "sequences and", total_bases, "nucleotides"

                # Store predictions in file
                with open(prediction_file, "a") as struct_out:
                    base_index = 0
                    # Go over all sequences in the dataset
                    for seq_index in range(len(lengths)):
                        # Get current sequence probabilities
                        seq_len = lengths[seq_index]
                        seq_probs , seq_comp_probs = list(), list()
                        for seq_base_index in range(seq_len):
                            seq_probs.append(predictions[base_index])
                            seq_comp_probs.append(1.0 - predictions[base_index])
                            base_index += 1
                        # Remove flanking areas (if necessary)
                        if flank > 0:
                            seq_probs = seq_probs[flank:-flank]
                            seq_comp_probs = seq_comp_probs[flank:-flank]
                        # Write current sequence probabilities into a new line(s) in the file
                        struct_out.write(">XXX\n")
                        for p in seq_probs:
                            struct_out.write(str(p) + " ")
                        struct_out.write("\n")
                        for p in seq_comp_probs:
                            struct_out.write(str(p) + " ")
                        struct_out.write("\n")
                        # Add zero probabilities to additional structure contexts (if needed)
                        if struct_num > 2:
                            for i in range(struct_num - 2):
                                struct_out.write("0.0 0.0\n")
