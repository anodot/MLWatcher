import json
import requests
import argparse
import time
import conf

# class Anodot_send_Api
# This class is responsible to send metrics to Anodot servers
class Anodot_send_Api():
    def __init__(self, token):
        self.anodot_domain = conf.ANODOT_DOMAIN
        self.full_url = "https://" + self.anodot_domain + "/api/v1/metrics?token=" + token + "&protocol=anodot20"

    # Analyzing the response coming back from Anodot
    def analyze_resp(self, resp):
            resp_info = {}

            if(resp.status_code != 200):
                resp_info['anodot_api_resp_status'] = 'FAILED'
                resp_info['status_code'] = resp.status_code
                resp_info['body_content'] = resp.content
            else:
                resp_info['anodot_api_resp_status'] = 'PASSED'
                resp_info['status_code'] = resp.status_code
                if(resp.content != ""):
                    parsed_json = json.loads(resp.content.decode())
                    body = json.dumps(parsed_json, indent=4, sort_keys=True)
                    resp_info['body_content'] = body
                else:
                    resp_info['body_content'] = None

            return resp_info

    # Send the data to Anodot
    def send_data(self, body):
        headers = {'Content-Type':'application/json'}
        resp = requests.post(self.full_url, headers=headers, data=body, timeout=120)
        resp_info = self.analyze_resp(resp)

        return resp_info

# print example of input file
def print_example():
    text = '''
{"properties":{"what":"cpu","server_id":"5","country":"us","target_type":"gauge","version":"1"},"tags":{},"timestamp":1475260200,"value":64.0},
{"properties":{"what":"revenue","partner_name":"Tesla","region":"apac",","target_type":"counter",
"version":"2"},"tags":{},"timestamp":1475260200,"value":7300.0},
{"properties":{"what":"revenue","partner_name":"Akamai","region":"us",","target_type":"counter",
"version":"2"},"tags":{},"timestamp":1475260200,"value":540.0}
    '''

    print(text)

def msg(name=None):
    return '''send_data.py
         Send data to Anodot based on Json file.

         - Get a Json example: python send_data.py --example
         - Send data to Anodot: python send_data.py --input <the Json input file> --env <prod/poc> --token <Anodot account toke>
        '''

def main():
    parser = argparse.ArgumentParser(description='', usage=msg())
    parser.add_argument('--input', required=False, default=None, help='The json input file path')
    parser.add_argument('--token', required=False, default=None, help="The Anodot's API token'")
    parser.add_argument('--example', required=False, action='store_true', help="Print example of the expected input file")

    args = parser.parse_args()
    if(args.example):
        print_example()
        exit()

    ano_api = Anodot_send_Api(args.token)
    with open(args.input, "r") as source_file:
        counter = 0
        sum_lines = 0
        content = "["
        for row in source_file:
            sum_lines += 1
            counter += 1

            if(counter == 900): # limit is up to 1000 data points in 1 http request.
                content += row.rstrip(",\n") + "]"
                counter = 0
                print(time.strftime("%c") + " " + str(ano_api.send_data(content)))
                content = "["
                time.sleep(0.375) # by default customer can send up to 2 http requests (2000 data points) per second
            else:
                content += row.rstrip("\n")
        # If last str in json is ',' delete it
        if content[-1] == ',':
            content = content[:-1] + "]"
        else:
            content += "]"
        print(time.strftime("%c") + " - " + str(ano_api.send_data(content)))
    source_file.close()

if __name__ == "__main__":
    main()