import sys

#Divide em pos ou neg de acordo com o label do arquivo rotulado passado
#como parâmetro por linha de comando

fname = sys.argv[1]

#file_label = fname + '_rotulado.txt'
file_label = 'all.txt' 
#path2 = 'test/'
#file_pos = path2 + fname + '_pos.txt'
#file_neg = path2+ fname + '_neg.txt'
path2 = 'train/TwoClasses/'
file_pos = path2 + 'pos/all_pos.txt'
file_neg = path2 + 'neg/all_neg.txt'


with open(file_label, 'r') as fl, open(file_pos, 'w') as fp, open(file_neg, 'w') as fn:
    for line in fl:
        line = line.rstrip()
        label = int(line.split(" ")[-1])
        content = ' '.join(line.split(' ')[:-1]) + '\n'

        if (label >= 1):
            fp.write(content)
        elif (label <= -1):
            fn.write(content)
