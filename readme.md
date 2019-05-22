# MLWatcher

MLWatcher is a python agent that records a large variety of time-serie metrics of your running ML classification algorithm.  
It enables you to monitor in real time :
- the predictions : monitor the repartition of classes, the distribution of the `predict_proba_matrix` values, anomalies in your predictions
- the features : monitor concept drift, anomalies in your data
- the labels : monitor accuracy, precision, recall, f1 of your predictions vs labels if applicable

The statistics derived from the data are :
- range, mean, std, median, q25, q50, q75, iqr for any continuous values (probabilities, features)
- count, frequency for any discrete values (labels, classes)

Some additional data are derived from the `predict_proba_matrix` and monitored as continuous values :
- pred1 : the maximum prediction for each line in the `predict_proba_matrix`
- spread12 : the spread between the maximum prediction(pred1) and the second maximum prediction for each line in the `predict_proba_matrix`.
Drops in the pred1 timeserie and jumps in the spread12 can indicate a decrease of the algorithm average degree of certainty of its predictions.

MLWatcher minimal input is the `predict_proba_matrix` of your algorithm (for each line in the batch of data, the probabilities of each class). 
`label_matrix` and `input_matrix` are optional to monitor.
In case of binary classification with only 2 classes, a threshold value can be fed to monitor the labels-related and prediction-related metrics.

## MLWatcher use cases

Monitoring your Machine Learning metrics can be used to achieve multiple goals: 
  - **alert on concept drift** : the production data can be significatively different from the training data as time passes by. Analyze the distribution of the production features through time (mean, std, median, iqr etc).  
    
  *Example of concept drift for MNIST dataset where the input pixels values get suddenly inverted. An anomaly in the distribution of the features is raised*:  
    
  ![Alt text](./IMAGES/concept_drift.png?raw=true "Concept drift")
    
  - **analyze the performance of the model** : the model may no longer be accurate with respect to the production data. Some unlabeled metrics can display this staleness : pred1, spread12, class_frequency. Drops in the pred1 and spread12 timeseries can indicate a decrease of the algorithm average degree of certainty of its predictions.  
    
  *Example of how the model predictions metrics change when a new set of input data comes into production*:   
    
  ![Alt text](./IMAGES/unlabeled_monitoring.png?raw=true "Model performance of predictions")
    
  - **check that your model is numerically stable** : analyze the pred1 and spread12 stability, but also the different classes frequency stability.  
    
  *Example of putting into production a weakly trained model (trained with a highly unbalanced training set) and how this affects the stability of the predictions distribution for production*:  
      
  ![Alt text](./IMAGES/class_distribution_anomaly.png?raw=true "Model numerical stability")
    
  - **canary process new models** : monitor multiple ML models with the production inputs and compare metrics of the production algorithm vs the tested ones, analyze the stability of each model through time, etc.   
    
  *Example of monitoring the accuracy metric for multiple concurrent algorithms*:  
      
  ![Alt text](./IMAGES/canary.png?raw=true "Canary process multiple models")
    
  - if labels are available, analyze the evolution of the classic ML metrics and correlate with other time series (features, predictions). 
  
The size of each buffer of data is also monitored, so it is important to also correlate the computed metrics with the sample size. (ie : **the sample size is not always statiscally significant**). 


## Getting Started

0- Install the libs in requirements.txt.
```
python -m pip install -r /path/to/requirements.txt
```
1- Add the MLWatcher folder in the same folder of your algorithm script.

2- Personalize some technical parameters in file conf.py (rotating logs specs, filenames,  token if applicable, etc).

3- Load the MLWatcher libs in your import lines :
```
from MLWatcher.agent import MonitoringAgent
```
4- Instanciate a MonitoringAgent object, and run the agent-server side:
```
agent = MonitoringAgent(frequency=5, max_buffer_size=500, n_classes=10, agent_id='1', server_IP='127.0.0.1', server_port=8000)
```
`frequency` :  (int) Time in minutes to collect data. Frequency of monitoring  
`max_buffer_size` : (int) Upper limit of number of inputs in buffer. Sampling of incoming data is done if limit is reached 
`n_classes` : (int) Number of classes for classification. Must be equal to the number of columns of your predict_proba matrix  
`agent_id` : (string) ID. Used in case of multiple agent monitors (default '1')  
`server_IP` : (string) IP of the server ('127.0.0.1' if local server)  
`server_port` : (int) Port of the server (default 8000)  

For LOCAL Server. Local server would be listening on previously defined port, on localhost interface (127.0.0.1).
```
agent.run_local_server()
```
For DISTANT Server : Hosted server would be listening on a defined port, on localhost interface (--listen localhost) or all interfaces (--listen all). Recommended :
```
python /path/to/server.py --listen all --port 8000 --n_sockets 5
```
See --help for server.py options.

5- Monitor the running ML process for each batch of data
```
agent.collect_data(
predict_proba_matrix = <your pred_proba matrix>,   ##mandatory
input_matrix = <your feature matrix>,  ##optional
label_matrix = <your label matrix>   ##optional
)
```
6- If `TOKEN`=None is provided in conf.py, you can analyze your data stored locally in the PROD folder with the given jupyter notebook (ANALYTICS folder)

7- **For advanced analytics of the metrics and detect anomalies in your data**, the agent output is compatible with Anodot Rest API by using a valid `TOKEN`.

You can use the Anodot API script as follows :

- Asynchronously from json files written on disk:
```
python anodot_api.py --input <path/to/PROD/XXX_MLmetrics.json> --token <TOKEN>
```
- Synchronously without storing any production file :
Edit `TOKEN`='123ad..dfg' instead of None in conf.py
In case the data is not correctly sent to Anodot(connection problems), the agent will start writing the metrics directly on disk in PROD folder.
Please contact Anodot to get a TOKEN through a trial session.

### Prerequisites

The agent is fully writen in Python 3.X. It was tested with Python >= 3.5

The input format for the agent collector are :  
predictions (mandatory): `predict_proba_matrix` size (batch_size x n_classes)  
labels (optional): `label_matrix` binary matrix of shape (batch_size x n_classes) or (int matrix of shape (batch_size x 1)  
features (optional) : `input_matrix` size (batch_size x n_features)  
n_classes must be >= 2

### Installing

See Getting Started section.
You can also have a look and run the example given with the MNIST dataset in the EXAMPLE folder (requirement:tensorflow).

## Deployment and technical features.

The agent structure is as follows:
  - a light agent-client side that collects the data of the algorithm and sends it the agent-server running in background (don't forget to launch agent.run_local_server() or use server.py)
  - a agent-server side that **asynchronously** handles the data received to compute a wide variety of time series metrics.
 

It skips the data if a problem is met and records logs accordingly. 
The agent is a light weight collector that stores up to `max_buffer_size` datapoints every period. 
Above this limit, sampling is done using a 'Reservoir sampling' algorithm so the sampled data remains statistically significant. 
 
To tackle bottleneck issues, you can adjust the number of threads that the server can run in parallel with the volume of batches you want to monitor synchronously. 
You can also adjust `max_buffer_size` and `frequency` parameters accordingly to your volumetry. 
For Anodot usage, a limit from Anodot API is defined as 2000 metric-datapoints per second. Please make sure that the volumetry is below this limit, else some monitored data would be lost (no storage case). 
Before going to production, a phase of **tests** for implementing the agent and server to your production running algorithm is **highly recommended**. 


## Contributing

This agent was developped by Anodot to help the data science community to monitor in real time the performance, the anomalies and the lifecycle of running ML algorithms.  
Please also refer to the paper of Google 'The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction' to have a global view of the good practices in production ML algorithm design and monitoring.

## Versioning

v1.0

## Authors

* **Anodot**
* **Garry B** - *ITC Project for Anodot*


## License

MIT License

Copyright (c) 2019 Anodot

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

Anodot Team  
Glenda  
