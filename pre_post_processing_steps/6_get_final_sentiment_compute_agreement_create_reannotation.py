from __future__ import division
from nltk import agreement
from scipy import stats
import numpy as np
import csv
import pickle
import pandas as pd

######## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! the order of paragraphs should be fixed before running this code !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import custom_distance as ds

# incase of smallset and largeset where the last column with label is the sentiment label
def get_final_sentiment(input_csv = 'smallset1.csv'):
    # input: csv file 
    # output: the last column of the csv as sentiment label of the document
    # warning: this method is just useful if there is no Fianl sentiment label in the document and the last column
    # of the csv is the final label with another name
    df = pd.read_csv(input_csv)
    df_out = df.ffill(axis=1).iloc[:, [-1]]
    return(df_out)


# get the array of the sentiments according to different annotators and return the label for the document
def getCombinedSentiment(sentiments):
    # input: array of the sentiments
    # if the variance is more than 2 it will be marked for reassignment to the new annotator
    # output: compute the sentiment based on the labels, return one final sentiment 
    weights = {'Negative': -2, 'Slightly_Negative': -1, 'Neutral': 0, 'Slightly_Positive': 1, 'Positive': 2}
    labels = ['Negative', 'Neutral', 'Positive']
    avg = 0
    sent = []
    kappa_input = [0,0,0]
    for s in sentiments:
        try:
            avg += weights[s]
            sent.append(weights[s])
            if 'Neg' in s:
              kappa_input[0] += 1
            elif 'Pos' in s:
              kappa_input[2] += 1    
            elif 'Neu' in s:
              kappa_input[1] += 1                      
        # skip nan labels
        except:
            continue
    var = np.var(sent)
    reannotate = False

    if var > 2:
        reannotate = True 
#         print('var is : %f'%var, 'sent is ' , sent)
    avg = float(avg)/float(len(sentiments))
    if avg >= 0.5:
        return (2,labels[2]),reannotate
    elif avg <= -0.5:
        return (0,labels[0]),reannotate
    else:
        return (1,labels[1]),reannotate


# preprocess sentences mostly used for titles, remove extra whitespace, <spand> tags for HTML multiple extra spaces
def preprocess(sent):
    # input: text to be processed
    # output: the removed whitespace version of the text with no <span> HTML tag
    return sent.replace('</span>','').replace('<span style="color:red;">','').replace(',','').replace('  ',' ').strip()
    
    
    # read all documents in crowdsource folder and keep the document text based on the title in hash
def create_hash(path,entity_file=True):
    import os
    # input: path: directory of file source ,entity_file: whether the first line of the file is entity name or not
    # output: a hash with key  of title and value of document
    title_text_hash= {}
    for file_name in os.listdir(path):
        document = open(path+file_name)
        # if we read from the documents where the first line is the entity name, we should skip it to get the title
        if (entity_file):
            ent = document.readline()
        document_title = document.readline()
        preprocessed_title = preprocess(document_title)
        text = document.read()
        title_text_hash[preprocessed_title,ent.strip()] = text.strip()
    return title_text_hash
    

def create_annotation_data(labels,global_task_id):
    data_annot = []
    for ind in range(len(labels)):
        data_annot.append((global_task_id*10+ind,global_task_id,frozenset([labels[ind]])))
    t2 = AnnotationTask(data=data_annot ,distance=ds('cs_3class.txt'))
    return t2.avg_Ao()     
    
    # load amazon turk asnwer file as csv and data frame and return documents, entitie, titles, class label
# Also it saves the pickle file of title and labels as hash to use for multiple submissions


# load amazon turk asnwer file as csv and data frame and return documents, entitie, titles, class label for each paragraph
# Also it saves the pickle file of title and labels as hash to use for multiple submissions
#also it computes pairwise agreement 


all_labels = []
all_labels = []
def mturk_output_creator_paragraph_level(input_csv = 'long_sentences_with_second_batch_submission_multipleSubmission.csv'
                                        ,path = 'LDC2014E13_output/crowdsource/',
                                        pickle_reader = True, save_file_name = 'alldata_3Dec.csv',
                                        num_labels=3):
    # input: pickle_reader: there is a hash file of title and labels saved as pickle file, if this variable is true
    # this hash is read and used for furthur usage, otherwise it will be created 
    # path: the path of the crowdsource files
    # header_id: for general csv file, we don't have index after column, for merged csv file, we have id of 2,
    # this parameter determines whether to add the index to the end of title and entity column or not
    # save_id: when we need to run multiple times and save multiple files, we use different indexing to prevent confusing
    # input_csv: input csv file which is the output of MTurk
    # output: a hash table of titles and different labels by the user
    # a csv file of the entity, doc index, title, sentiment and document text
#     title_text_hash = create_hash(path)
    final_labels_all = []
    para_labels_all = []
    title_column = 'Input.title'
    title_text_hash = create_hash(path)
    
    if pickle_reader:
        title_labels_hash = pickle.load(open('title_labels.pickle', 'rb'))
        doc_id = len(title_labels_hash)
    else:
        title_labels_hash = {}
        doc_id = 0
    columns = ['TARGET_ENTITY','DOCUMENT_INDEX','TITLE','DOCUMENT','TRUE_SENTIMENT','Paragraph0','Paragraph1'
                         ,'Paragraph2','Paragraph3','Paragraph4','Paragraph5' ,'Paragraph6','Paragraph7','Paragraph8',
                         'Paragraph9','Paragraph10','Paragraph11','Paragraph12','Paragraph13' ,'Paragraph14','Paragraph15','Reannotate','HITId']
        
    try:
        open(save_file_name)
        output_file = open(save_file_name, 'a+')
        fileWriter = csv.writer(output_file)
    except:
        output_file = open(save_file_name, 'a+')
        fileWriter = csv.writer(output_file)
        fileWriter.writerow(columns)
    data_df = pd.read_csv(input_csv)
    
    columns =[col for col in list(data_df) if col.startswith('Answer.sentiment_Pargraph')]
    print('ghabls az amal ', len(data_df))
    data_df = data_df[data_df['AssignmentStatus']== 'Approved']
    title = data_df[title_column]
    print('bad az amal ' , len(title))
    print('approved assignment %d out of %d assignments' %(len(data_df[data_df['AssignmentStatus']== 'Approved']),len(data_df)))
    entity = data_df['Input.entity']
    entity_decision= data_df['Answer.entity_decision']
    
#     all_doc_titles = title.unique()
    hitids = data_df['HITId'].unique()
    skipped_doc_entity = 0
    not_found_label_documents = 0
    
    print("number of documents %d" % len(hitids))
    
    global_id = 0
    try:
            Final_Sentiment = data_df[['Answer.Final_sentiment']] 
    except:
            Final_Sentiment = get_final_sentiment(input_csv)
    final_sentiment_column_name = list(Final_Sentiment)[0]
    Final_Sentiment = pd.concat([Final_Sentiment,data_df['HITId']],axis=1)
            
#     for item in all_doc_titles:
    for hid in hitids:
        item = list(data_df[data_df['HITId']==hid][title_column])[0]
        entity_decision_array = entity_decision[data_df[title_column]==item].tolist()
        document_entity = entity[data_df[title_column]==item].tolist()[0]
        
       #  if (item,document_entity) in title_labels_hash:
#             print('item is in the hash %s' % item)
#             continue
        
        
        final_entity_decision = max(set(entity_decision_array), key=entity_decision_array.count)
        # if most of the annotator mark the document as not related to the specified entity, skip it
        if final_entity_decision =='No':
            skipped_doc_entity += 1
            continue
        try:
            clean_title = preprocess(item)
        except:
            clean_title = preprocess(item.encode('utf-8'))
        try:
        
            document = title_text_hash[clean_title,document_entity]
# 			document = data_df[data_df['HITId']==hid][title_column]
        except:
            print(document_entity)
            print('title not found  ')
            continue
        row = [document_entity,doc_id,clean_title,document]
        df_allparagraphs = data_df[data_df['HITId']==hid]
        df_finalsentiment = Final_Sentiment[Final_Sentiment['HITId']==hid]
        
        final_labels = df_finalsentiment[final_sentiment_column_name].dropna().tolist()
        l,reannotate = getCombinedSentiment(final_labels)
        label = l[1]
        if len(final_labels) == num_labels  :
            final_labels_all.append(final_labels)
        
        title_labels_hash[(item,document_entity)] = final_labels



        row.append(label)
        write_to_file = True
        ### IF the last paragraph label is the document label, it doesn't exclude it, we should remove it (TO DO)
        for column in columns:
           
            labels = df_allparagraphs[column].dropna().tolist()

            if len(labels) > 0:
                all_labels.append(labels)
                if len(labels) > num_labels:
                    write_to_file = False
                    print('larger than expect4ed, %s'%item)
                if len(labels) < num_labels:
                    write_to_file = False
                    print('hid %s has %d labels for column %s and length of the final labels is %d' %(str(hid),len(labels),str(column),len(final_labels)  ) )
                l,reann = getCombinedSentiment(labels)
                label = l[1]
                if reann:
                    reannotate = True
                global_id += 1
                    
            else:
                label = 'NaN'
            row.append(label)
        row.append(reannotate)
        row.append(hid)


        if write_to_file:
            fileWriter.writerow(row)
        doc_id += 1
    with open('title_labels.pickle', 'wb') as title_labels_pickle:
        pickle.dump(title_labels_hash, title_labels_pickle) 
    output_file.close()
    print('skipped %d'%skipped_doc_entity)
    return final_labels_all,all_labels
                

def kapa_computer(new_final_labels,weighted=False, weights= None):
    table = 1 * np.asarray(new_final_labels)   #avoid integer division

    n_sub, n_cat =  table.shape

    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()
    #assume fully ranked
    assert n_total == n_sub * n_rat

    #marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    if weighted:
        table_weight = 1 * np.asarray(weights)     
        table2 = np.matmul(table , table_weight)
        table2 = np.multiply(table2,table)
    else:
        table2 = table * table
   
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.))
    p_mean = p_rat.mean()



    p_mean_exp = (p_cat*p_cat).sum()
  
    kappa = float(p_mean - p_mean_exp) / (1- p_mean_exp)

    
    return kappa
       
import statsmodels.stats.inter_rater as ir
import random
import statsmodels.stats.inter_rater as ir

### the method to return the array of number of votes per category for each item.
### it can come in 3 categories as pos, neut, neg or 5 categories, pos, slig pos, neut, slig neg, neg
def create_sent_arr(finals,number_class=3,number_voters=3):
    new_final_labels = []
    wrong_voters = 0
    for item in finals:
        new_l = []
        if len(item) !=  number_voters:
#             print(item)
            wrong_voters += 1
            continue
        negs = 0
        neus = 0
        pos = 0
        slneg = 0
        slpos = 0
        for l in item:
            if l == 'Negative': 
                negs += 1
            elif l == 'Slightly_Negative': 
                if number_class == 5:
                      slneg += 1
                else:
                      negs += 1           
            elif 'Neutral' in l:
                neus += 1
                if number_class == 2:
                      negs += 1
                      
                
            elif l == 'Positive': 
                pos += 1
            
            else: # slightly positive
                if number_class == 5:
                      slpos += 1
                else:
                      pos += 1   
        if number_class == 5:        
        	new_final_labels.append([negs,slneg,neus,slpos,pos])  
        elif number_class== 2:
        	new_final_labels.append([negs,pos])  
        else:
        	new_final_labels.append([negs,neus,pos])  
    print('wrong number of voters: %d'%wrong_voters)    
    return new_final_labels
    
    
## create reannotation input data to assign to new annotators in mturk
## there are two input files, the first one is the original annotation from the mturk output, 
## the second one is the final sentiment file in which it shows the whether each input needs to be reannotated or not
## based on the variance of the paragraph and document labels.
                    
def create_reannotate_mturk_input(original_csv='./mturk_out_Aug19_part2.csv',
								final_sentiment_csv = './deleteme.csv',
								outfile_resubmit='mturk_input_part2_v2.csv',
								outfile_accepted= 'mturk_out_accepted_part2_v2.csv'):
	
	final_sent = pd.read_csv(final_sentiment_csv)	
	original_docs = pd.read_csv(original_csv)	
	reann_docs = final_sent[final_sent['Reannotate']==True]

	out_df =  open(outfile_resubmit, 'w')
	fileWriter = csv.writer(out_df)
	fileWriter.writerow(['entity','title','content'])
	
	hids = 	reann_docs['HITId']


	for id in hids:

		entity 	= list(original_docs[original_docs['HITId']==id]['Input.entity'])[0]
		title 	= list(original_docs[original_docs['HITId']==id]['Input.title'])[0]
		doc 	= list(original_docs[original_docs['HITId']==id]['Input.content'])[0]	
		fileWriter.writerow([entity,title,doc])
		
	accepted_docs = final_sent[final_sent['Reannotate']==False]
	out_df =  open(outfile_accepted, 'w')
	fileWriter = csv.writer(out_df)
	fileWriter.writerow(['entity','title','content'])
	
	hids = 	reann_docs['HITId']


	for id in hids:

		entity 	= list(original_docs[original_docs['HITId']==id]['Input.entity'])[0]
		title 	= list(original_docs[original_docs['HITId']==id]['Input.title'])[0]
		doc 	= list(original_docs[original_docs['HITId']==id]['Input.content'])[0]	
		fileWriter.writerow([entity,title,doc])
		
	
	out_df.close()
		


path = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp_paragraph_seperated_Aug19/'        
part= 'part1'
save_path =  'mturk_final_sentiemnt_%s.csv' % part
original_csv='./data/mturk_out_Aug19_%s.csv' % part


	
number_class = 5
weighted = True
w_cons = 45*np.pi/180

try:
	with open('final_labels__.out','rb') as f:                                 
		final_labels=pickle.load(f)
	with open('par_lables__.out','rb') as f:       
		par_labels=pickle.load(f)

# 	final_labels = []
# 	par_labels = []
# 	
# 	final6,all_labels = \
# 	mturk_output_creator_paragraph_level(input_csv = 'testi.csv' ,
# 	                                    path='../dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp_paragraph_seperated_aug19/',
# 	                                    pickle_reader=False,num_labels=3)
# 
# 
# 	final_labels.extend(final6)
# 	par_labels.extend(all_labels)

except Exception as e:
    print('error occurred: %s'%str(e))
    final_labels = []
    par_labels = []
    final_labels,all_lables = mturk_output_creator_paragraph_level(input_csv = original_csv,path=path,pickle_reader=False,save_file_name =save_path)
    final6,all_labels = mturk_output_creator_paragraph_level(input_csv = '../dataAnalysis/mturk_out/Mturk_dataset_for_crowdsource/mergeset.csv',
 	                                    path='../dataAnalysis/LDC2014E13_output/used_documents_in_MTurk_merge/small/',
 	                                    pickle_reader=False,
 	                                    save_file_name = 'all_data_aug19.csv')
    final_labels.extend(final6)
    par_labels.extend(all_labels)
 
    for i in [1,2,3,4,5]:
 	    final,all_labels  = \
 	    mturk_output_creator_paragraph_level(input_csv = '../dataAnalysis/mturk_out_1Dec/emnlp_%s.csv'%str(i),
 	                                        path='../dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp/',
 	                                             pickle_reader=True,save_file_name = 'all_data_aug19.csv')
 	    final_labels.extend(final)
 	    par_labels.extend(all_labels)
 
 
    final8,all_labels  = \
    mturk_output_creator_paragraph_level(input_csv = '../dataAnalysis/mturk_out/emnlp_PS_out.csv',
 	                                    path='../dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp_paragraph_seperated/',
 	                                     pickle_reader=True,save_file_name = 'all_data_aug19.csv')
    final_labels.extend(final8)
    par_labels.extend(all_labels)
 
    final9,all_labels  = \
    mturk_output_creator_paragraph_level(input_csv = '../dataAnalysis/mturk_out/kbp_mturk_out.csv',
 	                                    path='../dataAnalysis/LDC2014E13_output/used_documents_in_MTurk_KBP/',
 	                                    pickle_reader=True,
 	                                    save_file_name = 'all_data_aug19.csv')
    final_labels.extend(final9)
    par_labels.extend(all_labels)
 
    final10,all_labels  = \
    mturk_output_creator_paragraph_level(input_csv = '../dataAnalysis/mturk_out/Mturk_dataset_for_crowdsource/smallset1.csv',
 	                                    path='../dataAnalysis/LDC2014E13_output/used_documents_in_MTurk_merge/small/',pickle_reader=True,
 	                                    save_file_name = 'all_data_aug19.csv')
    final_labels.extend(final10)
    par_labels.extend(all_labels)
 
    final11,all_labels  = \
    mturk_output_creator_paragraph_level(input_csv = '../dataAnalysis/mturk_out/Mturk_dataset_for_crowdsource/largeset1.csv',
 	                                    path='../dataAnalysis/LDC2014E13_output/used_documents_in_MTurk_merge/large/',pickle_reader=True,
 	                                    save_file_name = 'all_data_aug19.csv')
 
    final_labels.extend(final11)
    par_labels.extend(all_labels)
 
 
    final12,all_labels  = \
    mturk_output_creator_paragraph_level(input_csv ='../dataAnalysis/mturk_out_1Dec/long_1.csv',
 	                                    path='../dataAnalysis/LDC2014E13_output/used_documents_in_MTurk_merge/large/',pickle_reader=True,
 	                                    save_file_name = 'all_data_aug19.csv')
    final_labels.extend(final12)
    par_labels.extend(all_labels)
 
 
    final13,all_labels  = \
    mturk_output_creator_paragraph_level(input_csv = '../dataAnalysis/mturk_out_4Dec/emnlp_batch2_mturk_out.csv',
 	                                    path='../dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp_paragraph_seperated_batch2/'
 	                                     ,pickle_reader=True,
 	                                     save_file_name = 'all_data_aug19.csv')
    final_labels.extend(final13)
    par_labels.extend(all_labels)
    final14,all_labels  = mturk_output_creator_paragraph_level(input_csv = '../dataAnalysis/mturk_out_4Dec/emnlp_batch3_mturk_out.csv',
 	                                    path='../dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp_paragraph_seperated_batch3/'
 	                                     ,pickle_reader=True,save_file_name = 'all_data_aug19.csv')
    final_labels.extend(final14)
    par_labels.extend(all_labels)

#### This is the documents before re-assignment
# 	final14,all_labels  = \
# 	mturk_output_creator_paragraph_level(input_csv = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_preparation_process/data/mturk_out_aug19_part1.csv',
# 										path='../dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp_paragraph_seperated_aug19/'
# 										 ,pickle_reader=False,save_file_name = 'all_data_aug19.csv')
# 
# 	final_labels.extend(final14)
# 	par_labels.extend(all_labels) 	
# 	
# 		
# 	final14,all_labels  = \
# 	mturk_output_creator_paragraph_level(input_csv = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_preparation_process/data/mturk_out_aug19_part2.csv',
# 										path='../dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp_paragraph_seperated_aug19/'
# 										 ,pickle_reader=False,save_file_name = 'all_data_aug19.csv')
# 
# 	final_labels.extend(final14)
# 	par_labels.extend(all_labels) 
# 						
#### part 3 is not reassigned, so we have just one version of it			
    final14,all_labels  = \
    mturk_output_creator_paragraph_level(input_csv = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_preparation_process/data/mturk_out_aug19_part3.csv',
										path='../dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp_paragraph_seperated_aug19_part2/'
										 ,pickle_reader=False,save_file_name = 'all_data_aug19.csv')

    final_labels.extend(final14)
    par_labels.extend(all_labels)

#### This is the documents after re-assignment. 3 out of 4 are selected which are the least diverse 
    final14,all_labels  = \
    mturk_output_creator_paragraph_level(input_csv = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_preparation_process/mturk_out_combined_versions_part1.csv',
										path='../dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp_paragraph_seperated_aug19/'
										 ,pickle_reader=False,save_file_name = 'all_data_aug19.csv')

    final_labels.extend(final14)
    par_labels.extend(all_labels)
	
    final14,all_labels  = \
    mturk_output_creator_paragraph_level(input_csv = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_preparation_process/mturk_out_combined_versions_part2.csv',
										path='../dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp_paragraph_seperated_aug19/'
										 ,pickle_reader=True,save_file_name = 'all_data_aug19.csv')

    final_labels.extend(final14)
    par_labels.extend(all_labels)
    print(len(final_labels))
    print(len(par_labels))
   
    with open('final_labels.out','wb') as f:
		pickle.dump(final_labels,f)
    with open('par_lables.out','wb') as f:
		pickle.dump(par_labels,f)
	

if number_class== 5:    
# 	weights = np.ones([5,5])                             
    weights = [[1,np.cos(w_cons/2),np.cos(w_cons),np.cos(w_cons*3/2),0], # negative
               [np.cos(w_cons/2),1,np.cos(w_cons/2),np.cos(w_cons),np.cos(w_cons*3/2)],  #Slightly Negative
               [np.cos(w_cons),np.cos(w_cons/2),1,np.cos(w_cons/2),np.cos(w_cons)], 	 #Neutral
               [np.cos(w_cons*3/2),np.cos(w_cons),np.cos(w_cons/2),1,np.cos(w_cons/2)],  #Slightly Positive
               [0,np.cos(w_cons*3/2),np.cos(w_cons),np.cos(w_cons/2),1]]  #Positive
else:
# 	weights = np.ones([3,3]) 
    weights = [[1				,np.cos(w_cons)	,0],
               [np.cos(w_cons)	,1				,np.cos(w_cons)],
               [0				,np.cos(w_cons)	,1]] 

######## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! the order of paragraphs should be fixed before running this code !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
print('number of paragraphs: ',len(   par_labels))
                                 
arr = create_sent_arr(par_labels,number_class,number_voters=3)
print('length of the arr is: %d'%len(arr))
agg_par = kapa_computer(arr, weighted=weighted,weights=weights)
print('paragraph level agreement number is : %s'%str(agg_par))

                                 
print('number of documents: ',len(   final_labels))      
arr = create_sent_arr(final_labels,number_class,number_voters=3)
agg_doc = kapa_computer(arr, weighted=weighted, weights=weights)
print('document level agreement number is : %s'%str(agg_doc))
                                     
# # create_reannotate_mturk_input(original_csv=original_csv,
# # 							final_sentiment_csv = save_path,
# # 							outfile_resubmit='mturk_input_%s_v2.csv'%part,
# # 							outfile_accepted= 'mturk_out_accepted_%s_v2.csv'%part)
