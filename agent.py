import socket
import os
import logging
import numpy as np
import datetime
import subprocess
import json
import time
import random
import MLWatcher.conf as conf
from MLWatcher.sanity_check import check_format
from MLWatcher.timer import RepeatedTimer


class MonitoringAgent:
    """
    Generate a new Monitoring Agent.
    An agent can:
    - run local server to compute the metrics asynchronously --> method run_local_server()
    - collect data from algorithm and send data in buffer format --> method collect_data()

    frequency : (int) Time in minutes to collect data. Frequency of monitoring.

    max_buffer_size : (int) Upper limit of number of inputs in buffer. Sampling of incoming data is done if limit is reached.
    
    n_classes : (int) Number of classes for classification. Must be equal to the number of columns of your predict_proba matrix.

    agent_id : (string) ID. Used in case of multiple agent monitors.

    server_IP : (string) IP of the server ('127.0.0.1' if local server)

    server_port : (int) Port of the server (default 8000)


    """
    def __init__(self, frequency, max_buffer_size, n_classes, agent_id='1', server_IP='127.0.0.1', server_port=8000):
        self._server_IP = server_IP
        self._server_port = server_port
        self._frequency = int(frequency * 60)
        self._max_buffer_size = max_buffer_size
        self._n_classes = n_classes
        self._agent_id = agent_id
        self._num_sampled = 0
        self._threshold = 0.5
        self._started = False

        self._num_buffer = 0
        self._timestamp = None
        self._timer = None

        self._input_matrix = None
        self._predict_proba_matrix = None
        self._label_matrix = None

        conf.setup_logger('log1', os.path.join(conf.LOGS_DIRNAME, conf.LOGS_AGENT_FILENAME + '_'+self._agent_id+'.log'),
                          conf.LOG_AGENT_MAX_SIZE, conf.LOG_AGENT_MAX_BACKUPS)
        self._log = logging.getLogger('log1')
        self._log.propagate = False

    def collect_data(self, predict_proba_matrix, input_matrix=None, label_matrix=None, threshold=0.5):
        """
        Collects data directly from ML algorithm in the agent buffer.

        predict_proba_matrix: Probabilities matrix. Mandatory. Float matrix of shape (n_lines, n_classes).

        input_matrix: Feature matrix. Optional. Float matrix of shape (n_lines, n_features).

        label_matrix: Label matrix. Optional. Binary matrix of shape (n_lines, n_classes) or Int matrix of shape (n_lines, 1)

        """
        self._threshold = threshold

        if not self._started:
            self._timestamp = time.mktime(datetime.datetime.now().timetuple())
            self._run_timer()
            self._started = True

        # 1- Check format of data. If a problem is met, don't fill the buffer and display warning message in logs.
        try:
            check_msg, predict_proba_matrix, input_matrix, label_matrix = check_format(predict_proba_matrix, input_matrix, label_matrix, self._n_classes)
        except Exception as err:
            self._log.warning("WARNING: Agent ID : {}. error in parameters : {}".format(self._agent_id, err))
            return None

        if check_msg != "":
            self._log.warning("WARNING: Agent ID : {}. Format error in parameters : {}".format(self._agent_id, check_msg))
            return None

        # 2- Add and/or reservoir sample all new entries
        if self._num_buffer < self._max_buffer_size:
            # If buffer is not full, add all/part of incoming data
            if self._num_buffer + len(predict_proba_matrix) > self._max_buffer_size:
                # Add part of entries up to the max_size limit if too many new data
                filling_data_size = self._max_buffer_size - self._num_buffer
                tmp_output_matrix, tmp_output_matrix_2 = predict_proba_matrix[:filling_data_size], predict_proba_matrix[filling_data_size:]
                tmp_input_matrix, tmp_input_matrix_2 = (None, None) if input_matrix is None else (input_matrix[:filling_data_size], input_matrix[filling_data_size:])
                tmp_label_matrix, tmp_label_matrix_2 = (None, None) if label_matrix is None else (label_matrix[:filling_data_size], label_matrix[filling_data_size:])

                self._add_entries(tmp_output_matrix, tmp_input_matrix, tmp_label_matrix)
                # Reservoir sample the other data
                self._apply_reservoir_sample(tmp_output_matrix_2, tmp_input_matrix_2, tmp_label_matrix_2)
            else:
                # Add all entries if the max_size limit if not reached
                self._add_entries(predict_proba_matrix, input_matrix, label_matrix)
        else:
            # If buffer is already full, only reservoir sample all the incoming data
            self._apply_reservoir_sample(predict_proba_matrix, input_matrix, label_matrix)
        return None

    def _apply_reservoir_sample(self, incoming_predict_proba_matrix, incoming_input_matrix, incoming_label_matrix):
        """
        Replace some of the buffer values by the sampled valued.
        :return: None
        """
        for i, j in self._reservoir_sample(self._predict_proba_matrix, incoming_predict_proba_matrix):
            self._predict_proba_matrix[i, :] = incoming_predict_proba_matrix[j, :]
            if incoming_input_matrix is not None:
                self._input_matrix[i, :] = incoming_input_matrix[j, :]
            if incoming_label_matrix is not None:
                self._label_matrix[i, :] = incoming_label_matrix[j, :]

    def _reservoir_sample(self, reservoir_data, arriving_data):
        """
        Returns a list of tuples for sampling and replacing the arriving data into the reservoir
        Reservoir sampling allows all samples to enter the reservoir with the same probability (unbiaised)
        Early data have highest probability to enter but highest probability to leave the reservoir
        Late data have lowest probability to enter but lowest probability to leave the reservoir
        :param reservoir_data: stored data
        :param arriving_data: new incoming data to be sampled and stored
        :return: [(index_reservoir to delete i, index_arriving_data to store j)]
        """
        result = []
        r = len(reservoir_data)
        n = len(arriving_data)
        for j in range(n):
            i = random.randint(0, r + self._num_sampled + j)
            if i < r:
                result.append((i, j))
        self._num_sampled += n
        return result

    def _add_entries(self, predict_proba_matrix, input_matrix, label_matrix):
        """
        Add ML entries to buffer
        :param predict_proba_matrix:
        :param input_matrix:
        :param label_matrix:
        :return: None
        """
        if self._predict_proba_matrix is None:

            self._predict_proba_matrix = predict_proba_matrix
            self._input_matrix = input_matrix
            self._label_matrix = label_matrix
            self._num_buffer = predict_proba_matrix.shape[0]

        else:
            self._predict_proba_matrix = np.append(self._predict_proba_matrix, predict_proba_matrix, axis=0)
            if self._input_matrix is not None:
                self._input_matrix = np.append(self._input_matrix, input_matrix, axis=0)
            if self._label_matrix is not None:
                self._label_matrix = np.append(self._label_matrix, label_matrix, axis=0)
            self._num_buffer += predict_proba_matrix.shape[0]


    def _send_to_server(self, input_matrix, predict_proba_matrix, label_matrix, timestamp, threshold, agent_id, n_classes):
        """
        Sends current buffer info to server for asynchronous computation of metrics
        :return None
        """
        predict_proba_matrix = predict_proba_matrix[:self._num_buffer].tolist() if predict_proba_matrix is not None else predict_proba_matrix
        input_matrix = input_matrix[:self._num_buffer].tolist() if input_matrix is not None else input_matrix
        label_matrix = label_matrix[:self._num_buffer].tolist() if label_matrix is not None else label_matrix
        try:
            sock = socket.socket()
            data = {'input_matrix': input_matrix,
                    'predict_proba_matrix': predict_proba_matrix,
                    'label_matrix': label_matrix,
                    'timestamp': timestamp,
                    'threshold': threshold,
                    'agent_id': agent_id,
                    'n_classes': n_classes}
            sock.connect((self._server_IP, self._server_port))
            serialized_data = json.dumps(data).encode(encoding='utf-8')
            sock.sendall(serialized_data)
        except TimeoutError as err:
            self._log.warning("WARNING: Agent ID : {}. Server {} is not responding on port {}".format(self._agent_id, self._server_IP, self._server_port))
        finally:
            sock.close()
        return None

    def _reinit_buffer(self):
        """
        Once a buffer is sent to server, reinit the value for the next buffer.
        :return: None
        """
        self._num_buffer = 0
        self._num_sampled = 0

        self._input_matrix = None
        self._predict_proba_matrix = None
        self._label_matrix = None

    def _run_timer(self):
        """
        Launch _timer_function every  _frequency minutes
        :return: None
        """
        self._timer = RepeatedTimer(self._frequency, self._timer_function)

    def _timer_function(self):
        """
        Send the collected data to the server and reinit the buffer.
        :return:
        """
        try:
            self._timestamp = time.mktime(datetime.datetime.now().timetuple())
            self._send_to_server(self._input_matrix, self._predict_proba_matrix, self._label_matrix, self._timestamp,
                             self._threshold, self._agent_id, self._n_classes)
            if self._num_sampled > 0:
                self._log.info("INFO: Agent ID : {}. Buffer size collected: {}. Sampled data not collected : {}".format(self._agent_id, self._num_buffer, self._num_sampled))
            else:
                self._log.info("INFO: Agent ID : {}. Buffer size collected: {}".format(self._agent_id, self._num_buffer))

        except ConnectionRefusedError as err:
            self._log.warning("WARNING: Agent ID : {}. Batch of size {} could not be processed by the server : "
                              "check agent.run_local_server() was executed or "
                              "that the server {} is still running on port {} or "
                              "that the volume of data is not too large".format(self._agent_id, self._num_buffer, self._server_IP, self._server_port))

        finally:
            self._reinit_buffer()

    def run_local_server(self, n_sockets=5):
        """
        Run LOCALLY the server side of the agent asynchronously. For distant server, run the server.py script on the distant server.
        n_sockets: Number of requests the server can run in parallel. Adaptable to the volume of agents and buffers you want to monitor asynchronously.
        :return: None
        """
        # Check if server port is already opened
        sock = socket.socket()
        result = sock.connect_ex((self._server_IP, self._server_port))
        if result == 0:
            self._log.info('INFO: Agent ID : {}. MLWatcher is already running on (ip, port) = {}.'.format(self._agent_id, (self._server_IP, self._server_port)))
            sock.close()
        else:
            try:
                self._p = subprocess.Popen([conf.PYTHON_PATH,
                                            os.path.join(conf.SERVER_DIRNAME, conf.SERVER_FILENAME),
                                            '--listen', 'localhost',
                                            '--port', str(self._server_port),
                                            '--n_sockets', str(n_sockets)])
                self._log.info('INFO: Agent ID : {}. Server was launched by agent and is now running LOCALLY on (ip, port) = {}. Server ready.'.format(self._agent_id, ('127.0.0.1', self._server_port)))

            except OSError as e:
                self._log.warning("WARNING: Server side execution failed:", e)
            except KeyboardInterrupt:
                self._log.info('INFO: MLWatcher has stopped running on (ip, port) = {}. Keyboard interrupt.'.format(
                    (self._server_IP, self._server_port)))
                self._p.kill()
            except SystemExit:
                self._log.info('INFO: MLWatcher has stopped running on (ip, port) = {}. System Exit.'.format(
                    (self._server_IP, self._server_port)))
                self._p.kill()
        return None
