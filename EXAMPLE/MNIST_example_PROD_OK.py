
# IMPORT EXTERNAL LIBRARIES
import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from time import sleep
from random import randint
import random
import pickle

#### ADD #####
# IMPORT AGENT LIB
from MLWatcher.agent import MonitoringAgent

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(WORKING_DIR, "MODEL_MULTICLASS")
MODEL_W_DIR = os.path.join(WORKING_DIR, "MODEL_MULTICLASS_WRONG")

if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

if not os.path.exists(MODEL_W_DIR):
        os.mkdir(MODEL_W_DIR)

##EDITABLE##
TRAIN_PHASE = False
WRONG_TRAIN = False
##END EDITABLE##

TRAIN_BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_FEATURES = 784
LEARNING_RATE = 1e-3

TEST_BATCH_SIZE = 15


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=name)


def calculate_model(wrong_model=False):

    # Download the dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Define placeholders

    # correct labels
    y_ = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='y_')
    # input data
    x = tf.placeholder(tf.float32, shape=(None, NUM_FEATURES), name='x')

    # Build the net

    hidden_size1 = 500
    hidden_size2 = 100

    W_fc1 = weight_variable((NUM_FEATURES, hidden_size1), name='W_fct1')
    b_fc1 = bias_variable((1, hidden_size1), name='b_fct1')
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    W_fc2 = weight_variable((hidden_size1, hidden_size2), name='W_fct2')
    b_fc2 = bias_variable((1, hidden_size2), name='b_fct2')
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_fc3 = weight_variable((hidden_size2, NUM_CLASSES), name='W_fct3')
    b_fc3 = bias_variable((1, NUM_CLASSES), name='b_fct3')

    y = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3, name='OP_TO_RESTORE')

    # define the loss function
    y_log = tf.log(y)
    cross_entropy = tf.reduce_sum(-1 * y_log * y_)

    # define Optimizer
    Optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(2500):
            input_images, correct_predictions = mnist.train.next_batch(TRAIN_BATCH_SIZE)
            input_images_test, correct_predictions_test = mnist.test.next_batch(TRAIN_BATCH_SIZE)

            if wrong_model == False:
                sess.run(Optimizer, feed_dict={x: input_images, y_: correct_predictions})
            else:
                # Train the 'wrong' model on altered images
                input_images_test_altered, correct_predictions_altered = modify_test_set(input_images,
                                                            correct_predictions,
                                                            modification='do_unbalanced_data',
                                                            batch_size=TRAIN_BATCH_SIZE)
                sess.run(Optimizer, feed_dict={x: input_images_test_altered, y_: correct_predictions_altered})

            if i % 128 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: input_images, y_: correct_predictions})
                print("step {}, training accuracy {}".format(i, train_accuracy))
                test_accuracy = sess.run([accuracy], feed_dict={x: input_images_test, y_: correct_predictions_test})
                print("Validation accuracy: {}.".format(test_accuracy))

        if wrong_model == False:
            saver.save(sess, MODEL_DIR+'/my_test_model_multiclass')
        else:
            saver.save(sess, MODEL_W_DIR + '/my_test_model_multiclass')


def modify_test_set(input_set, label_set, modification='brutally_inverse_pixels', batch_size=TEST_BATCH_SIZE):
    if modification == 'brutally_inverse_pixels':
        input_set_modified = np.array([1 - x for x in input_set])
    if modification == 'divide_brightness':
        input_set_modified = np.array([x * 0.50 for x in input_set])
    if modification == 'none':
        input_set_modified = input_set
    if modification == 'do_unbalanced_data':
        if random.random() >= 0.05:
            input_set_modified = input_set[np.array([np.argmax(x) not in [0, 2, 4, 6, 8] for x in label_set], dtype=bool), :]
            label_set_modified = label_set[np.array([np.argmax(x) not in [0, 2, 4, 6, 8] for x in label_set], dtype=bool), :]
            return input_set_modified, label_set_modified
        else:
            input_set_modified = input_set[np.array([np.argmax(x) not in [1, 3, 5, 7, 9] for x in label_set], dtype=bool), :]
            label_set_modified = label_set[np.array([np.argmax(x) not in [1, 3, 5, 7, 9] for x in label_set], dtype=bool), :]
            return input_set_modified, label_set_modified
    if modification == 'remove_high_greys':
        thresh = 0.9
        input_set_modified = np.array([[pixel if pixel < thresh else 0 for pixel in x] for x in input_set])
    if modification == '4_corner_black':
        n = 5
        input_set = np.copy(input_set)
        input_set_reshaped = np.reshape(input_set, (batch_size, 28, 28))
        input_set_reshaped[:, :n, :n] = 0
        input_set_reshaped[:, 28 - n:, :n] = 0
        input_set_reshaped[:, :n, 28 - n:] = 0
        input_set_reshaped[:, 28 - n:, 28 - n:] = 0
        input_set_modified = np.reshape(input_set_reshaped, (batch_size, 28 * 28))
    if modification == 'center_black':
        input_set = np.copy(input_set)
        input_set_reshaped = np.reshape(input_set, (batch_size, 28, 28))
        input_set_reshaped[:, 12:17, 12:17] = 0
        input_set_modified = np.reshape(input_set_reshaped, (batch_size, 28 * 28))
    if modification == 'translate':
        input_set = np.copy(input_set)
        input_set_reshaped = np.reshape(input_set, (batch_size, 28, 28))
        input_set_reshaped[:, 4:, :] = input_set_reshaped[:, :24, :]
        input_set_reshaped[:, :4, :] = 0
        input_set_modified = np.reshape(input_set_reshaped, (batch_size, 28 * 28))

    return input_set_modified


def do_predictions():

    # Load the dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Load the previously saved model and graph in sess
    sess = tf.Session()
    sess_W = tf.Session()
    # Let's load meta graph and restore input var x
    saver = tf.train.import_meta_graph(MODEL_DIR+'/my_test_model_multiclass.meta')
    saver_W = tf.train.import_meta_graph(MODEL_W_DIR+'/my_test_model_multiclass.meta')
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
    saver_W.restore(sess_W, tf.train.latest_checkpoint(MODEL_W_DIR))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")

    # Predict batches of data
    ##EDITABLE##
    I = 50
    J = 60
    STOP = 60
    agent = MonitoringAgent(frequency=1, max_buffer_size=100, n_classes=10, agent_id='MNIST_Multiclass', server_IP='127.0.0.1', server_port=8000)
    agent.run_local_server(n_sockets=5)
    ##END EDITABLE##

    while True:
        for i in range(STOP):
            batch_size = randint(1, 20)
            input_images_test, correct_predictions_test = mnist.test.next_batch(batch_size)
            input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='none', batch_size=batch_size)
            #input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='brutally_inverse_pixels', batch_size=batch_size)
            # input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='divide_brightness', batch_size=batch_size)
            # input_images_test_altered, correct_predictions_test = modify_test_set(input_images_test, correct_predictions_test, modification='do_unbalanced_data', batch_size=batch_size)
            # input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='remove_high_greys', batch_size=batch_size)
            # input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='4_corner_black', batch_size=batch_size)
            # input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='center_black', batch_size=batch_size)
            # input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='translate', batch_size=batch_size)

            if i >= I and i <= J and not WRONG_TRAIN:
                input_images_test = input_images_test_altered

            # result of the softmax : predict_proba matrix of size (batch_size, num_classes)
            if i >= I and i <= J and WRONG_TRAIN:
                output_proba = sess_W.run('OP_TO_RESTORE:0', feed_dict={x: input_images_test})
            else:
                output_proba = sess.run('OP_TO_RESTORE:0', feed_dict={x: input_images_test})


            ##HERE IS THE HOOK
            agent.collect_data(predict_proba_matrix=output_proba, input_matrix=input_images_test, label_matrix=correct_predictions_test)
            #agent.collect_data(predict_proba_matrix=output_proba, label_matrix=correct_predictions_test)
            #agent.collect_data(predict_proba_matrix=output_proba, input_matrix=input_images_test)
            #agent.collect_data(predict_proba_matrix=output_proba)

            sleep_time = randint(1, 10)
            sleep(sleep_time)


def main():
    if TRAIN_PHASE:
        calculate_model(wrong_model=WRONG_TRAIN)
    else:
        do_predictions()

def save_pickle(result, pickle_name):
    """Saves an object in pickle object file for future use"""
    with open(os.path.join(WORKING_DIR, pickle_name), 'wb') as file:
        pickle.dump(result, file)
        return True


if __name__ == "__main__":
    # execute only if run as a script
    main()
