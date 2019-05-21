"""
The server python file is responsible for launching the server side in listen mode and gives orders to the worker
when data are received:
   - calculate_metrix of the data
   - create a dictionnary of all metrics (for json style export)
   - if there is a token in conf.py, export the metrics to Anodot API, else to a json file in PROD

Those server side computations are run asynchronously for each data input flow.
The server can be launched locally with the method agent.run_local_server().
If you want to host the server, this script has to be directly launched with cmd (see --help)
"""

import sys
import socket
import argparse
from worker import calculate_metrix
from worker import export_to_API_dict, write_json, send_to_anodot
from sanity_check import check_format
import conf
import json
import os
import logging

conf.setup_logger('log3', os.path.join(conf.LOGS_DIRNAME, conf.LOGS_SERVER_FILENAME),
                  conf.LOG_SERVER_MAX_SIZE, conf.LOG_SERVER_MAX_BACKUPS)
log = logging.getLogger('log3')
log.propagate = False

def server_listen(listen, port, n_sockets):
    "Run listen mode for the server on all interfaces. Asynchronously handle inputs received."
    s = socket.socket()
    s.bind((listen, port))
    s.listen(n_sockets)
    log.info('INFO: Server is now running on (local ip, port) = {}. Server ready.'.format(('0.0.0.0' if listen == '' else '127.0.0.1', port)))
    while True:
        try:
            c, a = s.accept()
        except KeyboardInterrupt:
            log.warning("WARNING: Server was interrupted by user")
        except SystemExit:
            log.warning("WARNING: Server was interrupted")
        data = b''
        while True:
            block = c.recv(conf.BUFFER_SIZE)
            if not block:
                break
            data += block
        c.close()
        if len(data) > 0:
            try:
                unserialized_input = json.loads(data.decode(encoding='utf-8'))
                unserialized_input = [unserialized_input[x] for x in conf.SERVER_API_FORMAT]
                input_matrix, output_matrix, label_matrix, timestamp, threshold, agent_id, n_classes = unserialized_input
                check_msg, output_matrix, input_matrix, label_matrix = check_format(output_matrix, input_matrix, label_matrix, n_classes)
                if check_msg != "":
                    log.warning("WARNING: Agent ID : {}. Format error in received buffer of size {} : {}".format(agent_id, len(output_matrix), check_msg))
                else:
                    log.info('INFO: Server received buffer of size {}. Origin Agent: {}'.format(len(output_matrix), agent_id))
                    input_metrix, output_metrix, label_metrix, timestamp = calculate_metrix(input_matrix, output_matrix, label_matrix, timestamp, threshold)
                    d = export_to_API_dict(input_metrix, output_metrix, label_metrix, timestamp)
                    if conf.TOKEN is not None:
                        send_to_anodot(d, conf.TOKEN, agent_id)
                    else:
                        write_json(d, agent_id)
            except Exception as err:
                    log.warning("WARNING: Error in receiving buffer : {}".format(err))





def main():
    parser = argparse.ArgumentParser(description='MLWatcher server')
    parser.add_argument('--listen', required=False, default='all', choices=['all', 'localhost'], help='Listen interface : "localhost" for 127.0.0.1, "all" for 0.0.0.0. Default : all')
    parser.add_argument('--port', required=False, default=8000, help="Port where the server side is run. Default : 8000")
    parser.add_argument('--n_sockets', required=False, default=5, help="Number of sockets listening on the server side. Default : 5")

    args = parser.parse_args()
    listen, port, n_sockets = args.listen, int(args.port), int(args.n_sockets)
    if listen == 'all':
        listen = ''
    server_listen(listen, port, n_sockets)


if __name__ == "__main__":
    main()
