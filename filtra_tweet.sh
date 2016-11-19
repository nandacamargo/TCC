#!/bin/bash

#April 17
#This script is meant to clear the tweets collected, extracting only the text part

while IFS= read -r LINE || [[ -n "$LINE" ]]; do     
    #Saves the content of LINE in extracted, from the tag text": forward
    extracted="${LINE#*'text":'}" 
    #echo $extracted

    #Saves in extracted the previous content of extracted, but removing the part that goes
    #since source until the end of the string 
    extracted="${extracted%%,\"source*}"

    echo $extracted >> saida
done < stream_cinema.json

