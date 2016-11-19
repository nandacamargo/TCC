import os
import sys


files_list = ["10_Cloverfield_Lane__info.json", "Alice_Through_The_Looking_Glass__info.json", "Allegiant__info.json", "Angry_Birds__The_Movie__info.json", "Batman_v_Superman__info.json", "Captain_America__Civil_War__info.json", "Deadpool__info.json", "Finding_Dory__info.json", "Huntsman__info.json", "Jungle_Book__info.json", "Me_Before_You__info.json", "Ratchet_Clank__info.json", "Revenant__info.json", "Spotlight__info.json", "Truman__info.json", "Warcraft__The_Beginning__info.json", "X-Men__Apocalypse__info.json", "Zoolander__info.json", "Zootopia_info.json"]

size = len(files_list)
cont = 1

with open("all_info.json", "a") as outfile:
    outfile.write('{ \n "movies": [\n\t')
    for f in files_list:
        with open(f, 'r') as infile:
            lines = infile.readlines()
            lines[1] = lines[1].rstrip()
            if (cont < size):
                outfile.write(lines[1] + ",\n")
            else:
                outfile.write(lines[1] + "\n")
            cont+=1

    outfile.write(']}')
