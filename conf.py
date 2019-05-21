import os
import logging
from logging import handlers
import sys

#################
## DO NOT EDIT ##
#################

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIRNAME = WORKING_DIR
SERVER_FILENAME = 'server.py'
SERVER_API_FORMAT = ['input_matrix', 'predict_proba_matrix', 'label_matrix', 'timestamp', 'threshold', 'agent_id', 'n_classes']
PYTHON_PATH = sys.executable
TMP_PROD_FILENAME = 'tmp'
ANODOT_VERSION = '1'
ANODOT_DOMAIN = "api.anodot.com"
#ANODOT_DOMAIN = "api-poc.anodot.com"


##############
## EDITABLE ##
##############

#TOKEN

TOKEN = None
#TOKEN = "abcdxxxxx"

# TIMESTAMP

TIME_FORMAT = "%Y%m%d-%H%M%S_"

# PROD factory : where the json metrics are exported

PROD_DIRNAME = os.path.join(WORKING_DIR, "PROD")
if not os.path.exists(PROD_DIRNAME):
        os.mkdir(PROD_DIRNAME)
PROD_FILENAME = 'MLmetrics.json'
ROTATE_PROD_TIME = 3600    # Rotate production json every ROTATE_PROD_TIME seconds
BUFFER_SIZE = 8192

# LOGGING factory : logs of the Agent Monitoring and Anodot export to API if applicable (TOKEN is not None)

LOGS_DIRNAME = os.path.join(WORKING_DIR, "LOGS")
if not os.path.exists(LOGS_DIRNAME):
        os.mkdir(LOGS_DIRNAME)

LOGS_AGENT_FILENAME = 'Agent'        #Agent_<Agent_name>.log
LOG_AGENT_MAX_SIZE = 10*1024*1024    # 10 MB
LOG_AGENT_MAX_BACKUPS = 5

LOGS_SERVER_FILENAME = 'Server.log'
LOG_SERVER_MAX_SIZE = 10*1024*1024    # 10 MB
LOG_SERVER_MAX_BACKUPS = 5

LOGS_ANODOT_FILENAME = 'Anodot_API_Export.log'
LOG_ANODOT_MAX_SIZE = 50*1024*1024    # 50 MB
LOG_ANODOT_MAX_BACKUPS = 5


def setup_logger(logger_name, log_file, log_max_size, log_count, level=logging.INFO):

    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')

    # add a rotating handler
    rotationHandler = handlers.RotatingFileHandler(log_file, mode='a', maxBytes=log_max_size, backupCount=log_count)
    rotationHandler.setFormatter(formatter)

    # write only WARNING messages to console
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    streamHandler.setLevel(logging.WARNING)

    logger.setLevel(level)
    logger.addHandler(streamHandler)
    logger.addHandler(rotationHandler)

############