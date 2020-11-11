#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import spacy
nlp = spacy.load('en')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)


# In[3]:


#### Create a dataframe from the main dataset with two new columns:
#### enclose entity and its corefrences with <t> and </t> tags save in ENCLOSED_ENTITY column
#### replace all occurrences of main entity with [TGT] and save in MASKED_ENTITY column


def enclose_mask_entity(df):
    new_df = df
    docs = df['DOCUMENT']
    print(len(list(docs)))
    ind = -1
    enclosed_docs = []
    masked_docs = []
    for text in list(docs):
        if ind %100 == 0:
            print('document %d processed'%ind)

        ind +=1 
        doc_id = df['DOCUMENT_INDEX'].iloc[ind]
        entity = df['TARGET_ENTITY'].iloc[ind]

        sc = 0
        new_text_tag = ""
        new_text_mask = ""
        corefs = []
        
        try:
            ## preprocess for coreference resolution
            doc = nlp(text)

            # find all mentions and coreferences
            for item in doc._.coref_clusters:
            # if the head of cluster is the intended entity then add all of the mentions
                if entity in item.main.text or item.main.text in entity:
                    for span in item:
                        corefs.append(span)

            for item in sorted(corefs):  
                    
                    
                    pronoun = item.text
                    
                    ec = item.start_char

                    if ec < sc:
                        continue
                    
                    
                    new_text_tag += text[sc:ec] + '<T> '+pronoun+' </T>' 
                    new_text_mask += text[sc:ec] + ' [TGT] '
                    if '\n' in item.text:
                        new_text_mask += '\n'
                        new_text_tag += '\n'
                    sc = item.end_char
                           
            if len(corefs) >0:
                new_text_tag += text[sc:]
                new_text_mask += text[sc:]

            else:
                new_text_tag = text.replace(entity,'<T> '+entity+' </T>' )
                new_text_mask = text.replace(entity,' [TGT] ' )
#                 print('coref couldnt find the main entity in document %d'%doc_id)
        except Exception as e:
            print('can not resolve coref for document %d, error is %s '% (doc_id,str(e)))
            new_text_tag = text
            new_text_mask = text
        enclosed_docs.append(new_text_tag)
        masked_docs.append(new_text_mask)
    new_df['ENCLOSED_DOCUMENT'] = pd.Series(enclosed_docs)
    new_df['MASKED_DOCUMENT'] = pd.Series(masked_docs)
    return new_df
    
            
data_train = pd.read_csv('alldata_aug19_train.csv', encoding='latin-1')
new_df = enclose_mask_entity(data_train)  
new_df.to_csv('alldata_aug19_enclosed_masked_train.csv', encoding='latin-1')
# 
data_dev = pd.read_csv('alldata_aug19_dev.csv', encoding='latin-1')
new_df = enclose_mask_entity(data_dev)  
new_df.to_csv('alldata_aug19_enclosed_masked_dev.csv', encoding='latin-1')
# 
# 
data_test = pd.read_csv('alldata_aug19_random_test.csv', encoding='latin-1')
new_df = enclose_mask_entity(data_test)  
new_df.to_csv('alldata_aug19_enclosed_masked_random_test.csv', encoding='latin-1')
# 
# 
data_test_fixed = pd.read_csv('alldata_aug19_fixed_test.csv', encoding='latin-1')
new_df = enclose_mask_entity(data_test_fixed)  
new_df.to_csv('alldata_aug19_enclosed_masked_fixed_test.csv', encoding='latin-1')
# 



# # DataStats Statistics

# In[42]:


def data_dist(input_file):
    doc_lengths = []
    par_length_withDoc = []
    par_length_woDoc = []
    i=-1
    for doc in input_file['DOCUMENT']:
        i += 1
        length = len(doc.split(' '))
        doc_lengths.append(length)
        if pd.notnull(input_file['Paragraph0'].iloc[i]):
            pars = doc.split('\n')
            for par in pars:
                par_length_withDoc.append(len(par.split(' ')))
                par_length_woDoc.append(len(par.split(' ')))
        else:
            par_length_woDoc.append(length)
            
    return(doc_lengths,par_length_withDoc,par_length_woDoc)

            
    


# In[43]:


from matplotlib import pyplot
(doc_lengths,par_length_withDoc,par_length_woDoc)=data_dist(data_train)
pyplot.figure()
pyplot.hist(doc_lengths,bins=[0,128,256,384,512]);
pyplot.title('Distributaion of Document Length in Train Set')
pyplot.xlabel('Document Length')
pyplot.ylabel('Frequency')


pyplot.figure()
pyplot.hist(par_length_withDoc,bins=[0,20,40,60,80,100,120,140,160]);
pyplot.title('Distributaion of Paragraph Length in Train Set')
pyplot.xlabel('Paragraph Length')
pyplot.ylabel('Frequency')




# In[44]:


(doc_lengths,par_length_withDoc,par_length_woDoc)=data_dist(data_dev)

pyplot.figure()
pyplot.hist(doc_lengths,bins=[0,128,256,384,512]);
pyplot.title('Distributaion of Document Length in Dev Set')
pyplot.xlabel('Document Length')
pyplot.ylabel('Frequency')






pyplot.figure()
pyplot.hist(par_length_woDoc,bins=[0,20,40,60,80,100,120,140,160]);
pyplot.title('Distributaion of Paragraph Length in Dev Set')
pyplot.xlabel('Paragraph Length')
pyplot.ylabel('Frequency')

pyplot.show()

