import json
from py2neo import Graph, authenticate

#This script is intended to load relevant information previously
#collected from IMDB into Neo4J database

authenticate("localhost:7474", "neo4j", "suasenha")
graph = Graph()

path = "../dataClassified/train/FiveClasses/"
category = ['vpos', 'pos', 'neutral', 'neg', 'vneg']
files = ["Alice", "Deadpool", "Finding_Dory", "Huntsman", "Mogli", "Ratchet", "Revenant", "Warcraft", "War", "Zootopia"]
real_names = ["Alice Through the Looking Glass", "Deadpool", "Finding Dory", "The Huntsman: Winter's War", "The Jungle Book", "Ratchet & Clank", "The Revenant", "Warcraft: The Beginning", "Captain America: Civil War", "Zootopia"]


with open("tweets_per_movie.json", "w") as out_file:
    out_file.write('{ "tweets": [\n')
    for curr_class in category:
        i = 0
        path2 = path + curr_class + "/"
        for f in files:
            file = path2 + f + "_" + curr_class + ".txt"
            with open(file, 'r') as data_file:
                
                for line in data_file:
                    line = line.rstrip()
                    json = '{ "movie": "'+ real_names[i] + '", "text": ' + line + ', "category": "' + curr_class + '" },\n'
                    out_file.write(json)
            i+=1        

    out_file.write("]}")

with open("tweets_per_movie.json", "r") as data_file:
    json = json.load(data_file)

#Data collected using Twitter's Streaming API and preprocessed to keep only useful information
query = """
WITH {json} AS doc
UNWIND doc.tweets AS tweets

MERGE (t:Tweet {text: tweets.text}) 
MERGE (l:Label {name: tweets.category})
MERGE (t)-[:HAS]->(l)

WITH t, tweets
MATCH (m:Movie)
	WHERE m.title = tweets.movie
MERGE (t)-[:ABOUT]->(m)

"""


graph.run(query, json = json).dump()

