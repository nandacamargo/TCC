import json
import string
import time
import os


fname = 'data/movies_now.txt'
with open(fname, 'r') as f:
    for line in f:
        script = "nohup python3 streaming_downloader.py -q " + "'" + line.rstrip('\n') + "'" + " -d data &"
        os.system(script)
        time.sleep(600)
        #print("nohup -q " + "'" + line.rstrip('\n') + "'" + " -d data")
