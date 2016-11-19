import requests
import string
import json

def write_file(outfile, data, details):
        try:
            with open(outfile, 'a') as f:
                json.dump(data, f)
                f.write('\n')
                json.dump(details, f)
                
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

##########################################
#The two functions below are from:
#https://gist.github.com/bonzanini/af0463b927433c73784d

def format_filename(fname):
    #Convert file name into a safe string.
    return ''.join(convert_valid(one_char) for one_char in fname)


def convert_valid(one_char):
    #Convert a character into '_' if invalid.
    valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
    if one_char in valid_chars:
        return one_char
    else:
        return '_'


########################################

#File with the list of movies to collect info
fname = 'filmes_em_cartaz.txt'

with open(fname, 'r') as f:
    for line in f:
        url = 'http://www.omdbapi.com/?s=' + line
        resp = requests.get(url)
        if resp.status_code != 200:
            #Different of success: this means something went wrong.
            raise ApiError('GET movie {}'.format(resp.status_code))
        else:
            print("=====>\nColetando filme: " + line)
            search = resp.json()['Search']
            
            cont = 0     
            for item in search:
                #print('{} {}'.format(item['Title'], item['Year'])
                if ((item['Year'] == '2016' or item['Year'] == '2015') and item['Type'] == 'movie'):
                    
                    #In case there is more than one link to the same movie searched, cont will be the numeration of the output file
                    cont += 1
                    url_id = 'http://www.omdbapi.com/?i=' + item['imdbID']

                    resp2 = requests.get(url_id)
                    details = resp2.json()
            
                    #Removing special characters from the name of output file
                    # safe_line = format_filename(item['Title'])
                    safe_line = format_filename(line)
                    
                    if (cont > 1):
                        outfile = 'output_IMDB/' + safe_line + '_info' + str(cont) + '.json'
                    else:
                        outfile = 'output_IMDB/' + safe_line + '_info.json'

                    #Writing in the file
                    write_file(outfile, item, details)
