#Some initial tests
from nltk.corpus import wordnet as wn


movie = wn.synsets('movie')
print(movie)
print(movie[0].definition())
print(movie[0].examples())

movie1 = wn.synset('movie.n.01')
print(movie1.lemmas())

print(movie1.hypernyms())
print(movie1.hyponyms())

