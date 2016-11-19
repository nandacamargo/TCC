import sys
import os

#Rótulos:
# -2: muito negativo
# -1: negativo
#  0: neutro
#  1: positivo
#  2: muito positivo


#===================
#Importante

#Assume-se que a estrutura do arquivo lido é composta por tweet, espaço e rótulo
#===================

fname = '../label/allLabeled.txt'
path = '../label/'
path2 = 'train/FiveClasses/'

files_list = [path2 + 'vpos/all_vpos.txt', path2 + 'pos/all_pos.txt', path2 + 'neutral/all_neutral.txt', path2 + 'neg/all_neg.txt', path2 + 'vneg/all_vneg.txt']


with open(fname, 'r') as f, open(files_list[0], 'a') as all_vp, open(files_list[1], 'a') as all_p, open(files_list[2], 'a') as all_nn, open(files_list[3], 'a') as all_n, open(files_list[4], 'a') as all_vn:
    for file_label in f:
        file_label = file_label.rstrip()
        mainName = file_label.split('.')[0]
        mainName = '_'.join(mainName.split("_")[:-1])
        very_pos = path2 + 'vpos/' + mainName + '_vpos.txt'
        pos = path2 + 'pos/' + mainName + '_pos.txt'
        neutral = path2 + 'neutral/' + mainName + '_neutral.txt'
        neg = path2 + 'neg/' + mainName + '_neg.txt'
        very_neg = path2 + 'vneg/' + mainName + '_vneg.txt'

        file_label = path + file_label
        
        with open(file_label, 'r') as fl, open(very_pos, 'w') as vp, open(pos, 'w') as p, open(neutral, 'w') as nn, open(neg, 'w') as n, open(very_neg, 'w') as vn:
            for line in fl:
                line = line.rstrip()
                label = int(line.split(" ")[-1])
                content = ' '.join(line.split(' ')[:-1]) + '\n'
                   

                if (label == 2):
                    vp.write(content)
                    all_vp.write(content)
                elif (label == 1):
                    p.write(content)
                    all_p.write(content) 
                elif(label == 0):
                    nn.write(content)
                    all_nn.write(content)
                elif(label == -1):
                    n.write(content)
                    all_n.write(content)
                elif(label == -2):
                    vn.write(content)
                    all_vn.write(content)

             
