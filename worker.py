"""
The worker script receives orders from the server side.
It executes the following tasks that are done on the server commands :
   - calculate_metrix of the inputs
   - create a dictionnary of all metrics (for json style export)
   - if there is a token in conf.py, export the metrics to Anodot API, else to a json file in PROD
"""
import conf
from metrics import calculate_metrix_dicts
from anodot_api import Anodot_send_Api
from collections import defaultdict
import os
import json
import time
from decimal import Decimal
import datetime
import logging
import requests
import numpy as np


def rotate_prod_filename(tmp_filepath, tmp_filename, rotating_time):
    """
    Checks if the production file needs to be rotated: if last timestamp is > rotating_time, rotate, else don't
    :return: timestamp of production file to be written in
    """
    cur_timestamp = time.time()
    with open(os.path.join(tmp_filepath, tmp_filename), 'r') as file:
        old_timestamp = float(file.readline())
    if cur_timestamp >= old_timestamp + rotating_time:
        with open(os.path.join(tmp_filepath, tmp_filename), 'w') as file:
            file.write(str(cur_timestamp))
        return time.strftime(conf.TIME_FORMAT, time.localtime(cur_timestamp))
    else:
        return time.strftime(conf.TIME_FORMAT, time.localtime(old_timestamp))


def decimal_default(obj):
    """Used for writing a float as a float in the json export format"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError


def calculate_metrix(input_matrix, output_matrix, label_matrix, timestamp, threshold):
    """Transforms the inputs of the buffer client side, and calculate the dictionnary of metrics of each data matrix"""
    input_metrix, output_metrix, label_metrix = calculate_metrix_dicts(output_matrix,
                                                                       input_matrix,
                                                                       label_matrix,
                                                                       threshold)

    return input_metrix, output_metrix, label_metrix, timestamp


def export_to_API_dict(input_metrix, output_metrix, label_metrix, timestamp):
    """Adapt the metrics stored in dict to Anodot API format"""
    d = defaultdict(list)
    for key, values in list(input_metrix.items()):
        d[key].append([timestamp, values[0], values[1], values[2], values[3], values[4], values[5]])
    for key, values in list(output_metrix.items()):
        d[key].append([timestamp, values[0], values[1], values[2], values[3], values[4], values[5]])
    for key, values in list(label_metrix.items()):
        d[key].append([timestamp, values[0], values[1], values[2], values[3], values[4], values[5]])
    return d


def write_json(d_, agent_id):
    """Write to the rotating prod file the json of the metrics in Anodot API format"""
    tmp_prod_filepath = os.path.join(conf.PROD_DIRNAME, conf.TMP_PROD_FILENAME)
    if os.path.exists(tmp_prod_filepath):
        curr_prod_filename = rotate_prod_filename(conf.PROD_DIRNAME, conf.TMP_PROD_FILENAME, conf.ROTATE_PROD_TIME)
    else:
        curr_time = time.time()
        curr_prod_filename = time.strftime(conf.TIME_FORMAT, time.localtime(curr_time))
        with open(os.path.join(conf.PROD_DIRNAME, conf.TMP_PROD_FILENAME), 'w') as file:
            file.write(str(curr_time))

    with open(os.path.join(conf.PROD_DIRNAME, curr_prod_filename + conf.PROD_FILENAME), 'a', newline='') as fp:
        for i, (key, value) in enumerate(d_.items()):
            data = parse_dictionary_to_json_format(value[0], agent_id)
            json.dump(data, fp, default=decimal_default)
            fp.write(",\n")


def parse_dictionary_to_json_format(list_of_properties, agent_id):
    """
    Parse the metrics list. If a field in properties is empty "", then delete this empty field.
    :param list_of_properties:
    :return: dictionary for json format export
    """
    d = {'tags': {}}
    timestamp = list_of_properties[0]
    #unixtime = time.mktime(timestamp.timetuple())
    d["timestamp"] = int(timestamp)
    d["value"] = Decimal(str(list_of_properties[1]))
    d["properties"] = {"what": list_of_properties[2],
                       "class_name": list_of_properties[3],
                       "feat_name": list_of_properties[4],
                       "stat": list_of_properties[5],
                       "target_type": list_of_properties[6],
                       "agent_id": agent_id,
                       "ver": conf.ANODOT_VERSION}
    # delete empty strings in properties
    for k, v in list(d["properties"].items()):
        if v == '':
            del d["properties"][k]

    return d


def send_to_anodot(d_, token, agent_id):
    """
    Send the memory stored metrics in Anodot API json format to Anodot API
    If the connection is lost, start writing the production into PROD folder for future export.
    """
    conf.setup_logger('log2', os.path.join(conf.LOGS_DIRNAME, conf.LOGS_ANODOT_FILENAME),
                      conf.LOG_ANODOT_MAX_SIZE,
                      conf.LOG_ANODOT_MAX_BACKUPS)
    _log = logging.getLogger('log2')
    _log.propagate = False

    try:
        ano_api = Anodot_send_Api(token)
        counter = 0
        sum_lines = 0
        content = "["
        for i, (key, value) in enumerate(d_.items()):
            data_dict = parse_dictionary_to_json_format(value[0], agent_id)
            data_dict["value"] = decimal_default(data_dict["value"])
            sum_lines += 1
            counter += 1

            if (counter == 900):  # limit is up to 1000 data points in 1 http request.
                content += str(data_dict) + "]"
                counter = 0
                _log.info("INFO: " + str(ano_api.send_data(content)))
                content = "["
                time.sleep(0.375)  # by default customer can send up to 2 http requests (2000 data points) per second
            else:
                content += str(data_dict) + ","
        # If last str in content is ',' delete it
        if content[-1] == ',':
            content = content[:-1] + "]"
        else:
            content += "]"
        _log.info("INFO: " + str(ano_api.send_data(content)))
    except requests.exceptions.ConnectionError as connectError:
        _log.warning("WARNING: Connection with Anodot API server was lost : {}".format(connectError))
        _log.info("INFO: Writing metrics into json format for future export")
        write_json(d_, agent_id)
    return None
