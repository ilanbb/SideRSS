from model import *
import time
import glob
import sys

# Random generator initializers
tf.set_random_seed(1)
np.random.seed(1)

# Set model parameters
params = dict()
params['rna_bases'] = 4
params['rna_structures'] = 1
params['batch_size'] = 1
params['batch_limit'] = 25000
params['beta'] = 0.001
params['learning_rate'] = 0.00002
params['num_epochs'] = 2
params['stop_check_interval'] = 50
params['max_gradient_norm'] = 1
params['top_k'] = 10
params['conv_env_sizes'] = [3, 5, 9]
params['hidden_layers_sizes'] = [64, 128]

# Set convolutional layers data
conv_layer1 = {'input_channels': 1, 'output_channels': 256, 'size_x': 2*params['conv_env_sizes'][0]+1, 'size_y':params['rna_bases']}
conv_layer2 = {'input_channels': 1, 'output_channels': 128, 'size_x': 2*params['conv_env_sizes'][1]+1, 'size_y':params['hidden_layers_sizes'][0]}
conv_layer3 = {'input_channels': conv_layer2['output_channels'], 'output_channels': 64, 'size_x':2*params['conv_env_sizes'][2]+1, 'size_y':1}
conv_layers = [conv_layer1, conv_layer2, conv_layer3]

# Set strides
strides_x_y = (1, 1)

# Set network name
network_name = './siamese-network'

start_time = time.time()

if len(sys.argv) < 2:
    print "Usage: python <mode=train/eval/predict> mode_parameteres..."
    exit()

# Set directive
directive = sys.argv[1]

# Structure learning
if directive == "train":
    if len(sys.argv) != 4:
        print ("Usage: python main.py train <train_seq_file> <train_struct_path>")
        exit()
    print ("Training...")
    train_seq_file = sys.argv[2]
    train_struct_file = sys.argv[3]
    test_seq_file = None
    test_struct_file = None
    predictor = SideRSS(train_seq_file, test_seq_file, train_struct_file, test_struct_file, params, network_name, conv_layers, strides_x_y)
    predictor.train()

# Model evaluation
elif directive == "eval":
    if len(sys.argv) != 4:
        print ("Usage: python main.py eval <test_seq_file> <test_struct_path>")
        exit()
    print ("Evaluating...")
    train_seq_file = None
    train_struct_file = None
    test_seq_file = sys.argv[2]
    test_struct_file = sys.argv[3]
    predictor = SideRSS(train_seq_file, test_seq_file, train_struct_file, test_struct_file, params, network_name, conv_layers, strides_x_y)
    mse, mae, pr = predictor.batch_eval()
    print "MSE:", mse
    print "MAE:", mae
    print "PR:", pr

# Structure prediction
elif directive == "predict":
    if len(sys.argv) != 5:
        print ("Usage: python main.py predict <test_seq_file> <flank_size> <prediction_file>")
        exit()
    print ("Predicting...")
    train_seq_file = None
    train_struct_file = None
    test_seq_file = sys.argv[2]
    test_struct_file = None
    flank = int(sys.argv[3])
    prediction_file = sys.argv[4]
    predictor = SideRSS(train_seq_file, test_seq_file, train_struct_file, test_struct_file, params, network_name, conv_layers, strides_x_y)
    predictor.predict(prediction_file, flank)

else:
    print "Usage: python <mode=train/eval/predict> mode_parameteres..."
    exit()

end_time = time.time()
duration = end_time - start_time
print "Done:", duration, "seconds."
