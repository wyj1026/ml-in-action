import time
import requests

for i in range(100):
    requests.get("http://baidu.com")
    time.sleep(1)

