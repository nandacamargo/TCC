import json
import string
import sys
from geopy.geocoders import Nominatim

#Open a file with tweets and get the coordinates, if it's not null

geolocator = Nominatim()

file_list = ['stream_Alice.json', 'stream_Clank_2105.json', 'stream_deadpool0803.json', 'stream_deadpool1103.json', 'stream_deadpool.json', 'stream_Deadpool.json', 'stream_Finding_Dory.json', 'stream_Huntsman_2105.json', 'stream_Huntsman.json', 'stream_jungle_book.json', 'stream_Mogli_2105.json', 'stream_Mogli.json', 'stream_Ratchet_2105.json', 'stream_revenant_begin.json', 'stream_revenant.json', 'stream_War_2105.json', 'stream_Warcraft.json', 'stream_Zootopia_2105.json']


cont =1
for fname in file_list:
    data = []
    end_fname = fname.split('.')[0]
    infile = 'data/' + fname
    outfile = 'workflow/nLabel/info/' + end_fname + '.json'
    
    print("Processando file: " + infile)  
    with open(infile, 'r') as f, open(outfile, 'w') as out:
 
        nmbr_lines = 0
        out.write('{\n\t"type": "FeatureCollection",\n\t"features": [\n')
       
        for line in f:
            tweet = json.loads(line)
            if ("coordinates" in tweet.keys() and tweet.get('coordinates') != None and tweet['lang'] == 'en'):
                
                nmbr_lines+= 1
                if (nmbr_lines > 1): 
                    out.write(',\n')

                coord = tweet.get('coordinates')['coordinates']

                #If the coordinates are inside world boundaries
                if coord[0] >= -90 and coord[0] <= 90 and coord[1] >= -180 and coord[1] <= 180:
                    c = str(coord[1]) + ", " +  str(coord[0])
                    location = geolocator.reverse(c, timeout=100)
                    if (location.raw.get('address')):
                        if (location.raw['address'].get('country')):
                            country = location.raw['address']['country']
                            if (location.raw['address'].get('state')):
                                state = location.raw['address']['state']
                                address = state + ", " + country
                            else:
                                address = country
                
                else:
                    state = None
                    country = None
                    address = None
		
                info = {
                        "geometry": tweet['coordinates'],
                        "tweet_id": str(tweet['id']),
                        "tweet": tweet['text'],
                        "user_location": tweet['user']['location'],
                        "user_id": tweet['user']['id'],
                        "user_name": tweet['user']['name']
                }


                if (state is not None):
                    info['state'] = state

                if (country is not None):
                    info['country'] = country               
                    info['address'] = address 

                out.write(json.dumps(info, indent=4, separators=(',', ': ')))
		
        out.write('\n\t]\n}')

    print(str(cont) + " files concluídos")
    cont+=1
 
