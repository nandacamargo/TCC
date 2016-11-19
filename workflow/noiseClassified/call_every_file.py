import os
import sys

allLabeled = '../label/allLabeled.txt'

with open(allLabeled, 'r') as al:
    for file_label in al:
        file_label = file_label.rstrip()
        fname = ('_').join(file_label.split("_")[:-1])
        cmd = 'python3 noise_classification_info.py ' + fname
        print(cmd)
        os.system(cmd)