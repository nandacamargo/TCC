import json
from py2neo import Graph, authenticate

#This script is intended to load relevant information previously
#collected from IMDB into Neo4J database

authenticate("localhost:7474", "neo4j", "suasenha")
graph = Graph()

path = "../"
#files = ["Alice", "Deadpool", "Finding_Dory", "Huntsman", "Mogli", "Ratchet", "Revenant", "Warcraft", "War", "Zootopia"]
#real_names = ["Alice Through the Looking Glass", "Deadpool", "Finding Dory", "The Huntsman: Winter's War", "The Jungle Book", "Ratchet & Clank", "The Revenant", "Warcraft: The Beginning", "Captain America: Civil War", "Zootopia"]

files_list = ["../neg/NB/all_neg_NB.json", "../pos/NB/all_pos_NB.json"]

for fname in files_list:

	with open(fname, "r") as n:
	    json = json.load(n)

	print("Leu o arquivo")


    query2 = """
    WITH {json2} AS doc
    UNWIND doc.tweets AS prop

    FOREACH (x IN CASE WHEN prop.user_id IS NULL THEN [] ELSE [1] END |
    	MERGE (u:User {user_name : prop.user_name}) 
       		ON CREATE SET u.user_id = prop.user_id, u.user_location = prop.user_location)

    FOREACH (x IN CASE WHEN prop.tweet_id IS NULL THEN [] ELSE [1] END |    
    	MERGE (t:Tweet {tweet_id : prop.tweet_id}) 
    		ON CREATE SET t.text = prop.tweet)

    WITH prop
    MATCH (u:User), (t:Tweet)
        WHERE prop.user_name = u.user_name AND prop.tweet_id = t.tweet_id
    MERGE (u)-[:TWEETED]->(t)

    WITH t, prop
    MATCH (m:Movie)
    	WHERE m.title = prop.movie
    MERGE (t)-[:ABOUT]->(m)

    MERGE (l:Label {name: prop.label})
    MERGE (t)-[:CLASSIFIED_WITH_NB]->(l)

    FOREACH (state in prop.state |
        MERGE (s:State {name: prop.state}) ON CREATE
            SET s.country = prop.country
        MERGE (c:Country {name: prop.country})
        MERGE (c)-[:HAS]->(s)

    	MERGE (t)-[:FROM]-> (s)
    	MERGE (t)-[:FROM]-> (c))

    """

    graph.run(query2, json2 = json2)

    print("Inseridos os nós classificados pelo NB")