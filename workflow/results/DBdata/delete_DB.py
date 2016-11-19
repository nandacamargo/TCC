import json
from py2neo import Graph, authenticate

#This script is intended to delete all the information previously
#inserted into Neo4J database

authenticate("localhost:7474", "neo4j", "suasenha")
graph = Graph()


query = """
MATCH p=()-[f:FROM]->()  DELETE f
"""

graph.run(query).dump()

print("Relacionamento FROM deletado")
##################################

query = """
MATCH p=()-[a:ABOUT]->()  DELETE a
"""

graph.run(query).dump()

print("Relacionamento ABOUT deletado")
##################################

query = """
MATCH p=()-[h:HAS]->()  DELETE h
"""
graph.run(query).dump()
print("Relacionamento HAS deletado")
##################################

query = """
MATCH p=()-[t:TWEETED]->()  DELETE t
"""
graph.run(query).dump()
print("Relacionamento TWEETED deletado")
##################################

query = """
MATCH (c:Country), (s:State) DELETE c, s
"""

print("Nodes Country e State deletados")
##################################

query = """
MATCH (u:User) DELETE u
"""

print("Node User deletado")
##################################

query = """
MATCH (t:Tweet), (m:Movie) DELETE t, m
"""

print("Nodes Tweet e Movie deletados")

