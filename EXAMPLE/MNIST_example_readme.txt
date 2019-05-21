This example implement MLWatcher to the multi class classification dataset MNIST.
The agents monitor the 784 inputs distribution, the predict proba matrix output and the labels, for different scenarios explained below.

Prerequisites : 
MLWatcher requirements
tensorflow
python>=3.5

0- Train a correct model with script MNIST_example_TRAIN.py

The first model corresponds to the correct input data and correct model.

  0-0 Edit :
  TRAIN_PHASE = True
  WRONG_TRAIN = False

  0-1 Run the script MNIST_example_TRAIN.py
  It will save a correct trained model in folder MODEL_MULTICLASS

1- Train a wrong model with script MNIST_example_TRAIN.py

The second model corresponds to a model trained with not significant input data.

  1-0 Edit :
  TRAIN_PHASE = True
  WRONG_TRAIN = True

  1-1 Run the script MNIST_example_TRAIN.py
  It will save a wrongly trained model in folder MODEL_MULTICLASS_WRONG
  (By default, the wrongly trained model was trained with an unbalanced train dataset with less than 5% even numbers)


2- Monitor you running algorithms in productions

The example is a production MNIST scenario where images batches size vary between 1 and 20, the time rate varies between 1 and 10 secs.
For a better comprehension, it is advised to run those 3 scenarios in parallel (at the same time) and for a sufficient number of datapoints in time series(at least 20 datapoints).
run MNIST_example_PROD_OK.py
run MNIST_example_PROD_NOK_INPUTS.py
run MNIST_example_PROD_NOK_MODEL.py

3- Exploit your results in PROD folder and use the ANALYTICS notebook.

Remark :
Images datasets have a large number of features, in this case the monitoring of the 784 features may not be the best practice.
You can restrict the monitoring to the predict_proba and labels matrix only.

Details in code for 2:
2-1 Scenario 1 : Monitor every 1 minute the correct algorithm and the correct data
  TRAIN_PHASE = False
  WRONG_TRAIN = False
  [..]
  agent = MonitoringAgent(frequency=1, max_buffer_size=100, n_classes=10, agent_id='MNIST_Multiclass', server_IP='127.0.0.1', server_port=8000)
  [..]
  input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='none', batch_size=batch_size)
  # input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='brutally_inverse_pixels', batch_size=batch_size)

2-2 Scenario 2 : Monitor every 1 minute the correct algorithm with periodically wrong data. Wrong data come between I and J values.
  TRAIN_PHASE = False
  WRONG_TRAIN = False
  [..]
  agent = MonitoringAgent(frequency=1, max_buffer_size=100, n_classes=10, agent_id='MNIST_Multiclass_wrong_inputs', server_IP='127.0.0.1', server_port=8000)
  [..]
  # input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='none', batch_size=batch_size)
  input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='brutally_inverse_pixels', batch_size=batch_size)

2-3 Scenario 3 : Monitor every 1 minute the wrongly trained algorithm with correct data. Wrong model is applied between I and J values.
  TRAIN_PHASE = False
  WRONG_TRAIN = True
  [..]
  agent = MonitoringAgent(frequency=1, max_buffer_size=100, n_classes=10, agent_id='MNIST_Multiclass_wrong_inputs', server_IP='127.0.0.1', server_port=8000)
  [..]
  input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='none', batch_size=batch_size)
  # input_images_test_altered = modify_test_set(input_images_test, correct_predictions_test, modification='brutally_inverse_pixels', batch_size=batch_size)
