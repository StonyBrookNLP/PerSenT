import pandas as pd
import json
import random
import os

source_path = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_resource/EMNLP_Data_Junting'
train_input = source_path + '/content_df_test_filtered.csv'
title_train_input = source_path + '/titles/title_selected_sources_ids_with_targets_test.csv'




text_df = pd.read_csv(train_input, error_bad_lines=False,delimiter='\t')
title_df = pd.read_csv(title_train_input, error_bad_lines=False,delimiter='\t')

input_json = json.load(open(source_path + '/raw_docs.json'))
title_input = source_path + '/titles/title_selected_sources_ids_with_targets_train.csv'#'emnlp18_data/titles/title_selected_sources_ids_with_targets_train.csv'
title_df = pd.read_csv(open(title_input ), error_bad_lines=False,delimiter='\t')


data_path  = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/dataAnalysis/emnlp18_data/'
subdir= 'emnlp_paragraph_seperated_Aug19_part2/'



##select No_of_samples random documents
No_of_samples = 5000
doc_ids = random.sample(input_json.keys(), No_of_samples)
for index in doc_ids:
    
    
    for root, dirs, files in os.walk(data_path):
    	
    	if index in files:
    		print ("File exists")
    		continue

    # if the file exsits, don't create it again
    try:
        # if it has been chosen previously, don't choose it again
        out_file = open(data_path+subdir+str(index))
#         out_file = open('./masked_entity_lm/'+str(index))
        continue
    except:
        pass
    # select random documents from source file and read the main document
    paragraphs = input_json[index].split('\n\n')
    
    if len(paragraphs) < 3:
        continue
    
    title = title_df[title_df['docid'] == int(index)]
    try:
        main_title = title.iloc[0]['title']
        out_file = open(data_path+subdir+str(index),'w')
#         out_file = open('./masked_entity_lm/'+str(index),'w')
        out_file.write(main_title)
        out_file.write('\n')
        
    except: # this index is not in title file
        continue
    for paragraph in paragraphs:
        
        try:
            out_file.write(paragraph)
        except:
            out_file.write(paragraph.encode('utf-8'))
        out_file.write('\n')
    out_file.close()
        