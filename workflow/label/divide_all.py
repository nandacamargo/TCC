fname = 'tudo'
file_label = 'all.txt' 
file_adds = 'noise_train/' + fname + '_noise.txt'
file_not_adds = 'notNoise_train/' + fname + '_notNoise.txt'


with open(file_label, 'r') as fl, open(file_adds, 'w') as fa, open(file_not_adds, 'w') as fn:
    for line in fl:
        line = line.rstrip()
        label = int(line.split(" ")[-1])
        content = ' '.join(line.split(' ')[:-1]) + '\n'

        if (label == 5):
            fa.write(content)
        else:
            fn.write(content)
