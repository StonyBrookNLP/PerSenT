#-*- coding: utf-8 -*-
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import spacy
import neuralcoref
import os
import re
coref = spacy.load('en')
neuralcoref.add_to_pipe(coref)
st = StanfordNERTagger('/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/entityRecognition/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz',
					   '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/entityRecognition/stanford-ner-2018-02-27/stanford-ner.jar',
					   encoding='utf-8')
					   
					   
# find the most frequent entity in document and write them on file
import os


source = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/dataAnalysis/emnlp18_data/'
subdir = 'emnlp_paragraph_seperated_Aug19_part2'

### reads all the file in a subdirectory and process one by one
for filename in os.listdir(source+'coref_%s/'%subdir):
    
    ### if the entity recognition for on document is solved (it is saved in another direcotry) skip that document
    if os.path.exists(source +'pairentity_%s/'%subdir +filename):
        continue
    print(filename)
    text = open(source+'coref_%s/'%subdir + filename)
    # a dictionary for each document to find the entity
    allEntities = {}
    lines = text.read()
    lines = lines.replace('”','"').replace('’','\'')
    try:
        tokenized_text = word_tokenize(lines)
    except:
        tokenized_text = word_tokenize(lines.decode('utf-8'))
    classified_text = st.tag(tokenized_text)
    i = 0
    previous_entity = ("","")
    two_previous_entity = ("","")
#   read all entities in document and double check to keep the PERSON ones 
    for pair in classified_text:
            # if the entity is person, save it in hash
        if (pair[1] == 'PERSON' and previous_entity[1] != 'PERSON'):
            if pair[0] in allEntities:
                allEntities[pair[0]] = allEntities[pair[0]] +1
            else:
                allEntities[pair[0]] = 1
        elif (pair[1] == 'PERSON' and previous_entity[1] == 'PERSON' and two_previous_entity[1] != 'PERSON'):
            entity = previous_entity[0]+" "+pair[0]
            if entity in allEntities:
                allEntities[entity] = allEntities[entity] +1
            else:
                allEntities[entity] = 1
        elif (pair[1] == 'PERSON' and previous_entity[1] == 'PERSON' and two_previous_entity[1] == 'PERSON'):
            # then add the new pairs as new entity
            entity = two_previous_entity[0]+" "+ previous_entity[0]+" "+pair[0]
            if entity in allEntities:
                allEntities[entity] = allEntities[entity] +1
            else:
                allEntities[entity] = 1
        two_previous_entity = previous_entity
        previous_entity = pair
    if len(allEntities) ==0:
        print('no Entities in %s'%filename)
        continue
    sortedEntities =  sorted(allEntities, key=allEntities.get, reverse=True)
    maxEntity = sortedEntities[0]
    counted = 0
    for item in sortedEntities:
        if maxEntity in item:
            counted += allEntities[item]
            maxEntity = item
    # if number of entities is less than 3 do not save that document
    if counted <3:
        print('number of dominate entity %d'%counted)
        continue
    print(maxEntity)
    outfile = open(source +'pairentity_%s/'%subdir +filename,'w')
    try:
        outfile.write(maxEntity)
    except:
        outfile.write(maxEntity.encode('utf-8'))
    outfile.write('\n')
    outfile.write(''.join(lines))
    outfile.close()