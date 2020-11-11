import pandas as pd
data_train = pd.read_csv('/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_preparation_process/combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_reindex_train.csv', encoding='latin-1')
data_dev = pd.read_csv('/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_preparation_process/combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_reindex_dev.csv', encoding='latin-1')
data_test = pd.read_csv('/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_preparation_process/combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_reindex_random_test.csv', encoding='latin-1')
data_test_fixed = pd.read_csv('/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_preparation_process/combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_reindex_fixed_test.csv', encoding='latin-1')


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

            
from matplotlib import pyplot
#(doc_lengths,par_length_withDoc,par_length_woDoc)=data_dist(data_train)
#pyplot.figure()
#pyplot.hist(doc_lengths,bins=[0,150,300,450,600,750,900.1050],color='b',alpha=1,edgecolor='black');
#pyplot.title('Distribution of Document Length in Train Set')
#pyplot.xlabel('Document Length')
#pyplot.ylabel('Frequency')
#pyplot.show()


#pyplot.figure()
#pyplot.hist(par_length_withDoc,bins=[0,20,40,60,80,100,120,140,160],edgecolor='black');
#pyplot.title('Distribution of Paragraph Length in Train Set')# (including docs without paragraph label)')
#pyplot.xlabel('Paragraph Length')
#pyplot.ylabel('Frequency')
#pyplot.show()
 
 
# pyplot.figure()
# pyplot.hist(par_length_woDoc,bins=[0,20,40,60,80,100,120,140,160],edgecolor='black');
# pyplot.title('Distribution of Paragraph Length in Train Set')# (excluding docs without paragraph label)')
# pyplot.xlabel('Paragraph Length')
# pyplot.ylabel('Frequency')
# pyplot.show()
 
(doc_lengths,par_length_withDoc,par_length_woDoc)=data_dist(data_dev)

pyplot.figure()
pyplot.hist(doc_lengths,bins=[0,100,200,300,400,500,600,700,800,900,1000,1100,1200],edgecolor='black');
pyplot.title('Distribution of Document Length in Dev Set')
pyplot.xlabel('Document Length')
pyplot.ylabel('Frequency')
pyplot.show()

#
#pyplot.figure()
#pyplot.hist(par_length_withDoc,bins=[0,20,40,60,80,100,120,140,160],edgecolor='black');
#pyplot.title('Distribution of Paragraph Length in Dev Set')# (including docs without paragraph label)')
#pyplot.xlabel('Paragraph Length')
#pyplot.ylabel('Frequency')
#pyplot.show()
#
#
## pyplot.figure()
## pyplot.hist(par_length_woDoc,bins=[0,20,40,60,80,100,120,140,160],edgecolor='black');
### pyplot.title('Distribution of Paragraph Length in Dev Set')# (excluding docs without paragraph label)')
## pyplot.xlabel('Paragraph Length')
## pyplot.ylabel('Frequency')
## pyplot.show()
