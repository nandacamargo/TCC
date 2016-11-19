import json
import string
import sys

#Open a file with tweets; checks if the dict has lang and text fields in that line;
#If so and lang == 'en' writes on the output file.
# Command used to get the list of files: ls -la data |  awk '{print $9;}'


#Put here the list of files collected using Streaming API
file_list = ['stream_Alice.json', 'stream_angry_birds.json', 'stream_Batman_2105.json', 'stream_batman.json', 'stream_batman__superman.json', 'stream_civil_war.json', 'stream_Clank_2105.json', 'stream_Cloverfield.json', 'stream_Cloverfield_maio.json', 'stream_convergente.json', 'stream_Convergente.json', 'stream_deadpool0803.json', 'stream_deadpool1103.json', 'stream_deadpool.json', 'stream_Deadpool.json', 'stream_divergente.json', 'stream_Finding_Dory.json', 'stream_Ghostbusters.json', 'stream_Huntsman_2105.json', 'stream_Huntsman.json', 'stream_jungle_book.json', 'stream_Me_Before_You.json', 'stream_Mogli_2105.json', 'stream_Mogli.json', 'stream_O_Zoolander.json', 'stream_Ratchet_2105.json', 'stream_revenant_begin.json', 'stream_revenant.json', 'stream_Spotlight1103.json', 'stream_Spotlight_begin.json', 'stream_Superman_2105.json', 'stream_Truman.json', 'stream_WALL-E.json', 'stream_War_2105.json', 'stream_Warcraft.json', 'stream_X-Men.json', 'stream_Zootopia_2105.json']


cont = 1
for fname in file_list:
    data = []
    end_fname = fname.split('.')[0]
    infile = 'data/' + fname
    outfile = 'workflow/nLabel/' + end_fname + '_en.txt'
  
    print("Processando file: " + infile)  
    with open(infile, 'r') as f, open(outfile, 'w') as out:
	    for line in f:
	        data = json.loads(line)
	        if ("lang" in data.keys() and "text" in data.keys()):
                    if (data['lang'] == 'en'):
                        json.dump(data['text'], out)
                        out.write("\n")

    print(str(cont) + " files concluídos")
    cont+=1