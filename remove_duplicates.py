import sys
import re

#Remove lines that are exactly the same, except by the final link of tweet

if(len(sys.argv) < 2):
    print("Usage: python3 " + sys.argv[0] + " ARG1\n\tARG1 = file with lines to be removed")
    sys.exit(1)


lines_seen = []

infile = sys.argv[1]
outfile = infile.split('/')[1] 
outfile = 'rotulate/' + outfile.split('_')[0] + '_sem_repet.txt'

p = re.compile(r'(https:)')

#line is the collected line; line2 é the line seen, without the link (https://...)
with open(infile, 'r') as f1, open(outfile, 'w') as f2:
    for line in f1:
        sl = line.split(' ')
        line2 = line
        if (len(sl) > 2 and p.match(line)):
            #Gets the phrase until finds https:
            line2 = line[0:line.find('https:')]

        if line2.lower() not in lines_seen:
            f2.write(line)
            lines_seen.append(line2.lower())
