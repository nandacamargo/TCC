import sys
import os

#Se o rótulo for 1 ou 2 é positivo; se for -1 ou -2 é negativo

#===================
#Importante

#Assume-se que a estrutura do arquivo lido é composta por tweet, espaço e rótulo
#===================

fname = '../label/allLabeled.txt'
path = '../label/'
path2 = 'train/TwoClasses/'
pos = path2 + 'pos/all_pos.txt'
neg = path2 + 'neg/all_neg.txt'

with open(fname, 'r') as f, open(pos, 'w') as p, open(neg, 'w') as n:
    for file_label in f:
        file_label = path + file_label.rstrip()
        mainName = file_label.split('/')[-1]
        mainName = ('_').join(mainName.split("_")[:-1])
        print(mainName)
        file_pos = path2 + 'pos/' + mainName + '_pos.txt'
        file_neg = path2 + 'neg/' + mainName + '_neg.txt'

        with open(file_label, 'r') as fl, open(file_pos, 'w') as fp, open(file_neg, 'w') as fn:
            for line in fl:
                line = line.rstrip()
                label = int(line.split(" ")[-1])
                content = ' '.join(line.split(' ')[:-1]) + '\n'
         
                if (label == 1 or label == 2):
                    fp.write(content)
                    p.write(content)
                elif(label <= -1):
                    fn.write(content)
                    n.write(content) 
