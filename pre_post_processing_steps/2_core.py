# This Python file uses the following encoding: utf-8
# import en_coref_lg
import spacy
import neuralcoref
import os
import re
# coref = en_coref_lg.load()
coref = spacy.load('en')
neuralcoref.add_to_pipe(coref)
#source = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/dataAnalysis/LDC2014E13_output/'
source = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/dataAnalysis/emnlp18_data/'
# source = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/dataAnalysis/'
# subdir = 'crowdsource/'
# subdir = 'KBP/'
# subdir = 'emnlp/'
# subdir = 'emnlp_paragraph_seperated_batch3/'
# subdir = 'masked_entity_lm/'
subdir = 'emnlp_paragraph_seperated_Aug19_part2/'
# read all files in directory (the reports)
for filename in os.listdir(source+subdir):
	
# 	print(filename)
	textfile = open(source+subdir + filename)

	
	text = textfile.read()
	text = re.sub('[^0-9a-zA-Z.\n,!?@#$%^&*()_+\"\';:=<>[]}{\|~`]+', ' ', text)
	indd = 0
	if os.path.exists(source+'coref_%s/'%subdir+filename):
		continue
	try:
		doc = coref(text.decode('utf-8'))
		
	except Exception as e: 
		print('error occured: %s'%str(e))
		doc = coref(text)
# 		continue
	try:
		outputText = doc._.coref_resolved
	except:
		print('not being processed %s'%filename)
		continue

	indd = indd+1
	outputfile = open(source+'coref_%s/'%subdir+filename,'w')
	try:
		outputfile.write(outputText)
	except:
		outputfile.write(outputText.encode('utf-8'))
	outputfile.write('\n')
	outputfile.close()
	del doc

print('all coreferences found')        

print('output table created')





	