import json
from py2neo import Graph, authenticate

#This script is intended to load relevant information previously
#collected from IMDB into Neo4J database

authenticate("localhost:7474", "neo4j", "suasenha")
graph = Graph()

with open('all_info.json') as data_file:
    all_info = json.load(data_file)
    #all_info = data_file.readlines()
print(all_info)


#Data from JSON collected using OMDb API
query = """ 
WITH {all_info} AS doc
UNWIND doc.movies AS movies
foreach (movie in movies |
    MERGE (m:Movie {title: movie.Title}) ON CREATE
        SET m.genre = movie.Genre, m.imdbRating = movie.imdbRating, m.year = movie.Year, m.metascore = movie.Metascore, m.imdbVotes = movie.imdbVotes, m.awards = movie.Awards, m.language = movie.Language
    foreach (country in movie.Country |
        MERGE (c:Country {name: country})
        MERGE (m)-[:FROM]->(c)))
"""


graph.run(query, all_info = all_info).dump()
