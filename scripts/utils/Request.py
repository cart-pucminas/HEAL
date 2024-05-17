import requests 
import time

from config.ConfigCreateDatabase import config



def request_POST(url):
    qtdRequests = config.QTD_REQUEST
    timeRequests = config.TIME_REQUEST
    result = None
    for i in range(0, qtdRequests, 1):
        try:
            result = requests.post(url)
            if result:
                print("API integration SUCCESS...")
                break
            else:
                print("API integration ERROR... trying again...")
        except Exception:
            print("API integration ERROR... trying again...")
        time.sleep(timeRequests)
            
    return result