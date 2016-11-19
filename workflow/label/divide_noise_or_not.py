import sys
import os

#Se o rótulo for 5 é propaganda; caso contrário, é relacionado ao filme

#===================
#Importante
#Considerou-se que a estrutura do arquivo rotulado é:
# tweet, espaço mais rótulo ao final de cada linha.
#===================


with open('allLabeled.txt', 'r') as f:
    for file_label in f:
        file_label = file_label.rstrip()
        fname = file_label.split('.')[0]
        fname = fname.split("_")[0]
        file_adds = 'noise_train/' + fname + '_noise.txt'
        file_not_adds = 'notNoise_train/' + fname + '_notNoise.txt'
        print(file_adds)

        with open(file_label, 'r') as fl, open(file_adds, 'w') as fa, open(file_not_adds, 'w') as fn:
            for line in fl:
                line = line.rstrip()
                label = int(line.split(" ")[-1])
                content = ' '.join(line.split(' ')[:-1]) + '\n'
         
                if (label == 5):
                    fa.write(content)
                else:
                    fn.write(content)  
