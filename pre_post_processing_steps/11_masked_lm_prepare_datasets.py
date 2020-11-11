##### create mask data set from the main dataset
##### for each document, replace all occurences of the main entity with [TGT], for each document, replace
##### one of the [TGT] occurrences with [MASK] and add the new document to the new dataset with label TRUE
##### for each document, replace each occurrences of all other entities with [MASK] and add the new document
##### to the dataset with FALSE label.
import pandas as pd
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import time
st = StanfordNERTagger('../entityRecognition/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz',
                       '../entityRecognition/stanford-ner-2018-02-27/stanford-ner.jar',
                       encoding='utf-8')

def prepare_date(df):


    ### new dataframe has 3 columnds, the initial data document id, the masked document(new) and the new label (true or false)
    mask_lm_df = pd.DataFrame(columns=['DOCUMENT_INDEX','DOCUMENT','LABEL'])
    ind = -1

    now = time.time()
    for doc in list(df['MASKED_DOCUMENT']):
        if ind %100 == 0:
            print('document %d processed'%ind)
        true_docs = []
        ind += 1
        doc_id = df['DOCUMENT_INDEX'].iloc[ind]
        mask_count = doc.count('[TGT]')
        for i in range(1,mask_count+1):
            ## replace [TGT] one by one based on the occurrence number
            true_doc = doc.replace('[TGT]','[MASK]',i).replace('[MASK]','[TGT]',i-1)
            if not true_doc in true_docs:
                true_docs.append(true_doc)
        try:
            tokenized_text = word_tokenize(doc)
        except:
            tokenized_text = word_tokenize(doc.decode('utf-8'))
        classified_text = st.tag(tokenized_text)
        false_docs = []
        i = 0
        
        previous_entity = ("","")
        entity = ""
    #   read all entities in document and their entity tags
        for pair in classified_text:
   
                # if the entity is person, find the whole person name, replace it with mask add it to the false arrays and keep going
            if (pair[1] != 'PERSON' and previous_entity[1] == 'PERSON'):
                false_doc = doc.replace(entity,'[MASK]')
                if not false_doc in false_docs :
                    false_docs.append(false_doc)
            if (pair[1] == 'PERSON' and previous_entity[1] != 'PERSON'):
                entity = pair[0]
            elif (pair[1] == 'PERSON' and previous_entity[1] == 'PERSON'):
                entity += " "+ pair[0]

            previous_entity = pair
        ### add all documents in the false/true array to the data frame with False/True labels
        for item in true_docs:
            mask_lm_df = mask_lm_df.append({'DOCUMENT_INDEX':doc_id,'DOCUMENT': item,'LABEL':'TRUE'}, ignore_index=True)
        for item in false_docs:
            mask_lm_df = mask_lm_df.append({'DOCUMENT_INDEX':doc_id,'DOCUMENT': item,'LABEL':'FALSE'}, ignore_index=True)
            
    print('processing took %d seconds'%(time.time()-now))
    print(len(mask_lm_df[mask_lm_df['LABEL']=='TRUE']))
    print(len(mask_lm_df[mask_lm_df['LABEL']=='FALSE']))

    return mask_lm_df



##### load data set with MASKED_ENTITY column where main entities are replaced with [TGT]
#df_train = pd.read_csv('combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_reindex_train.csv', encoding='latin-1')
#mask_train = prepare_date(df_train)
#mask_train.to_csv('masked_lm/mask_lm_combined_shuffled_3Dec_7Dec_aug19_reindex_train.csv', encoding='latin-1')
#
#df_dev = pd.read_csv('combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_reindex_dev.csv', encoding='latin-1')
#mask_dev = prepare_date(df_dev)
#mask_dev.to_csv('masked_lm/mask_lm_combined_shuffled_3Dec_7Dec_aug19_reindex_dev.csv', encoding='latin-1')
#
df_rTest = pd.read_csv('combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_reindex_random_test.csv', encoding='latin-1')
mask_rTest = prepare_date(df_rTest)
mask_rTest.to_csv('masked_lm/mask_lm_combined_shuffled_3Dec_7Dec_aug19_reindex_random_test.csv', encoding='latin-1')

#df_fTest = pd.read_csv('combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_reindex_fixed_test.csv', encoding='latin-1')
#mask_fTest = prepare_date(df_fTest)
#mask_fTest.to_csv('masked_lm/mask_lm_combined_shuffled_3Dec_7Dec_aug19_reindex_fixed_test.csv', encoding='latin-1')
