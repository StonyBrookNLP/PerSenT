import pandas as pd
import collections
import random
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
def separate_train_test(path ):
    df = pd.read_csv(path)
    print('before dropping dupicate:' , len(df))
    df = df.drop_duplicates(subset='DOCUMENT', keep=False)
    print('after dropping dupicate:' , len(df))
# 	 dropped.to_csv('all_data_combined_shuffled_drop_dup_3Dec_7Dec_aug19_%s.csv'%str(set_name),index=False)



    entities = df['TARGET_ENTITY']
    # en_freq = collections.Counter(entities)
    unique_entities = entities.unique().tolist()
    # print(unique_entities)
    data_size = len(df)
    print('whole dataset size is %d'%data_size)
    # print(en_freq)
    # fixed test set
    # fixed_test_entities = ['Barack Obama', 'LeBron James', 'Hillary Clinton']
    fixed_test_entities = ['Donald Trump', 'Barack Obama','Trump','Obama']
    fixed_test_set = pd.DataFrame([],columns=['TARGET_ENTITY','DOCUMENT_INDEX','TITLE','DOCUMENT','TRUE_SENTIMENT'])
    for entity in fixed_test_entities:
        unique_entities.remove(entity)
        df_entity = df[df['TARGET_ENTITY'] == entity]
        fixed_test_set = fixed_test_set.append(df_entity)
    fixed_test_size = len(fixed_test_set)
    print('fixed test set size is  %d' %fixed_test_size)

    random_test_set = pd.DataFrame([],columns=['TARGET_ENTITY','DOCUMENT_INDEX','TITLE','DOCUMENT','TRUE_SENTIMENT'])
    random_test_size = len(random_test_set)
    random_test_entities = []
    # random test set
    while random_test_size < float(data_size)/10:
        # print(unique_entities)
        entity = random.choice(unique_entities)
        random_test_entities.append(entity)
        # print(entity)
        unique_entities.remove(entity)
        df_entity = df[df['TARGET_ENTITY']== entity]
        df_entity = df_entity.head(30)
        random_test_set = random_test_set.append(df_entity)
        random_test_size = len(random_test_set)
    print('random test set size is now %d'%random_test_size)

    dev_set = pd.DataFrame([],columns=['TARGET_ENTITY','DOCUMENT_INDEX','TITLE','DOCUMENT','TRUE_SENTIMENT'])
    dev_size = len(dev_set)
    dev_entities = []
    while dev_size < float(data_size) / 10:
        entity = random.choice(unique_entities)
        dev_entities.append(entity)
        unique_entities.remove(entity)
        df_entity = df[df['TARGET_ENTITY']== entity]
        df_entity = df_entity.head(30)
        dev_set = dev_set.append(df_entity)
        dev_size = len(dev_set)
    print('dev set size is now %d' % dev_size)
    train_set = pd.DataFrame([],columns=['TARGET_ENTITY','DOCUMENT_INDEX','TITLE','DOCUMENT','TRUE_SENTIMENT'])
    #enforce the train set to have just firt 30 documents of each entity
    for entity in unique_entities:
        df_entity = df[df['TARGET_ENTITY'] == entity]
        df_entity = df_entity.head(30)
        train_set = train_set.append(df_entity)

    # train_set = df[df['TARGET_ENTITY'].isin( unique_entities)]
    print('train set size is  %d' % len(train_set))
    return train_set,unique_entities,dev_set, dev_entities,random_test_set,random_test_entities,fixed_test_set,fixed_test_entities


def plot_class_distribution(dataset,negative,positive,neutral):
    print('distribution of %s set among three classes"'%dataset)
    print('negative:\n ', negative)
    print('positive:\n ', positive)
    print('neutral:\n ', neutral)
    # Data to plot
    plt.figure()
    labels = 'Negative', 'Positive', 'Neutral'
    sizes = [negative,
             positive
        , neutral]

    colors = ['Red', 'Green', 'Yellow']

    # Plot
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.title(dataset, horizontalalignment='center', verticalalignment='bottom')
    # plt.show()
    pylab.savefig('%s.png'%dataset)


def entity_frequency(df):
    plt.figure()
    entities = df['TARGET_ENTITY']
    entity_count = collections.Counter(entities)
    data = entity_count.most_common(10)
    plot_df = pd.DataFrame(data, columns=['entity', 'frequency'])
    plot_df.plot(kind='bar', x='entity')
    pylab.savefig('%s.png' % 'entity_frequency')

def paragraph_distribution(df,plot=True):
    # plot sentence information
    plt.figure()
    documents = df['DOCUMENT'].tolist()
    doc_length = []
    for document in documents:
        try:
            doc_length.append(len(document.split('\n')))
        except:
            continue
    if plot:
        plt.hist(doc_length,len(set(doc_length)))
        plt.xlabel('Number of Paragraphs')
        plt.ylabel('Frequency')
        plt.axis([0, 30, 0,  2000])
        pylab.savefig('%s.png' % 'paragraph_freq')
        # plt.legend()
        print('total number of sentences:%d'% sum(doc_length))
        print('plot done')
    return doc_length

def word_distribution(df,plot=True,plot_name='train_dev'):
    # plot sentence information
    plt.figure()
    documents = df['DOCUMENT'].tolist()
    sentence_length = []
    for document in documents:
        try:
            sentences = document.split('\n')

        except:
            continue
        for sentence in sentences:
            sentence_length.append(len(sentence.split()))
    if plot:
        plt.hist(sentence_length,len(set(sentence_length)))
        plt.xlabel('Number of Words in Sentence')
        plt.ylabel('Frequency')
        plt.axis([0, 120, 0, 5000])
        pylab.savefig('%s.png' % plot_name)
        # plt.legend()
        print('total number of sentences:%d'% sum(sentence_length))
        print('plot done')
    return sentence_length


# path = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/dataAnalysis/data/alldata_12Nov.csv'
path = 'all_data_aug19.csv'
train_set,train_entities,dev_set, dev_entities,random_test_set,random_test_entities,fixed_test_set,fixed_test_entities = separate_train_test(path)
train_set.to_csv('alldata_aug19_train.csv',index=False)
dev_set.to_csv('alldata_aug19_dev.csv',index=False)
random_test_set.to_csv('alldata_aug19_random_test.csv',index= False)
fixed_test_set.to_csv('alldata_aug19_fixed_test.csv',index= False)

print('train set is: \n',train_entities)
print('dev set is: \n',dev_entities)
print('random test set is: \n',random_test_entities)
print('fixed test set is: \n',fixed_test_entities)

print('number of unique entities in train set is %d' % len(train_entities))
print('number of unique entities in dev set is %d' %len(dev_entities))
print('number of unique entities in random test set is %d'%len(random_test_entities))
print('number of unique entities in fixed test set is %d'%len(fixed_test_entities))


#
#
plot_class_distribution('train',len(train_set[train_set['TRUE_SENTIMENT']=='Negative']),
                        len(train_set[train_set['TRUE_SENTIMENT'] == 'Positive']),
                        len(train_set[train_set['TRUE_SENTIMENT'] == 'Neutral']))

plot_class_distribution('dev',len(dev_set[dev_set['TRUE_SENTIMENT']=='Negative']),
                        len(dev_set[dev_set['TRUE_SENTIMENT'] == 'Positive']),
                        len(dev_set[dev_set['TRUE_SENTIMENT'] == 'Neutral']))

plot_class_distribution('random_test',len(random_test_set[random_test_set['TRUE_SENTIMENT']=='Negative']),
                        len(random_test_set[random_test_set['TRUE_SENTIMENT'] == 'Positive']),
                        len(random_test_set[random_test_set['TRUE_SENTIMENT'] == 'Neutral']))

plot_class_distribution('fixed_test',len(fixed_test_set[fixed_test_set['TRUE_SENTIMENT']=='Negative']),
                        len(fixed_test_set[fixed_test_set['TRUE_SENTIMENT'] == 'Positive']),
                        len(fixed_test_set[fixed_test_set['TRUE_SENTIMENT'] == 'Neutral']))


train_set= pd.read_csv('alldata_aug19_train.csv')
dev_set = pd.read_csv('alldata_aug19_dev.csv')
random_test_set=  pd.read_csv('alldata_aug19_random_test.csv')
fixed_test_set = pd.read_csv('alldata_aug19_fixed_test.csv')
all_used_docs = train_set.append(dev_set).append(random_test_set).append(fixed_test_set)
plot_class_distribution('fixed_test',len(fixed_test_set[fixed_test_set['TRUE_SENTIMENT']=='Negative']),
                        len(fixed_test_set[fixed_test_set['TRUE_SENTIMENT'] == 'Positive']),
                        len(fixed_test_set[fixed_test_set['TRUE_SENTIMENT'] == 'Neutral']))
# all_used_docs.to_csv('/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/dataAnalysis/data/alldata_12Nov_30Nov.csv',index= False)
# entity_frequency(all_used_docs)
paragraph_distribution(all_used_docs,plot=True)

para_train = paragraph_distribution(train_set,plot=False)
para_dev = paragraph_distribution(dev_set,plot=False)
para_fixed_test = paragraph_distribution(fixed_test_set,plot=False)
para_random_test = paragraph_distribution(random_test_set,plot=False)
print('paragraphs in train: %d, dev %d, fixed test %d, random test %d' %(sum(para_train),sum(para_dev),sum(para_fixed_test),sum(para_random_test)))
print('max paragraphs in train: %d, dev %d, fixed test %d, random test %d' %(max(para_train),max(para_dev),max(para_fixed_test),max(para_random_test)))

word_distribution(all_used_docs,plot=True)

sent_train = word_distribution(train_set.append(dev_set),plot=False)
sent_dev = word_distribution(dev_set,plot=False)
sent_fixed_test = word_distribution(fixed_test_set,plot=False)
sent_random_test = word_distribution(random_test_set,plot=False)
print('words in train: %d, dev %d, fixed test %d, random test %d' %(sum(sent_train),sum(sent_dev),sum(sent_fixed_test),sum(sent_random_test)))
print('max wordsin train: %d, dev %d, fixed test %d, random test %d' %(max(sent_train),max(sent_dev),max(sent_fixed_test),max(sent_random_test)))


