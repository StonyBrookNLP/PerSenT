import pandas as pd
import collections
import random
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

train_set = pd.read_csv('./combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_train.csv')
dev_set = pd.read_csv('./combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_dev.csv')
random_test_set =  pd.read_csv('./combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_random_test.csv')
fixed_test_set  =  pd.read_csv('./combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_fixed_test.csv')

def count_entities(df):
	entities = df['TARGET_ENTITY']
	unique_entities = entities.unique().tolist()
	return unique_entities

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
        plt.axis([0, 25, 0,  4000])
        pylab.savefig('%s.png' % 'paragraph_freq')
        # plt.legend()
        print('total number of sentences:%d'% sum(doc_length))
        print('plot done')
    return doc_length

def word_distribution(df,plot=True,plot_name='train_words'):
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
        plt.xlabel('Number of Words in a Sentence')
        plt.ylabel('Frequency')
        plt.axis([0, 120, 0, 7500])
        pylab.savefig('%s.png' % plot_name)
        # plt.legend()
        print('total number of sentences:%d'% sum(sentence_length))
        print('plot done')
    return sentence_length


train_set = pd.read_csv('./combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_train.csv')
dev_set = pd.read_csv('./combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_dev.csv')
random_test_set =  pd.read_csv('./combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_random_test.csv')
fixed_test_set  =  pd.read_csv('./combined_set/all_data_combined_shuffled_3Dec_7Dec_aug19_fixed_test.csv')

train_entities = count_entities(train_set)
dev_entities = count_entities(dev_set)
random_test_entities = count_entities(random_test_set)
fixed_test_entities = count_entities(fixed_test_set)

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


all_used_docs = train_set.append(dev_set).append(random_test_set).append(fixed_test_set)
plot_class_distribution('fixed_test',len(fixed_test_set[fixed_test_set['TRUE_SENTIMENT']=='Negative']),
                        len(fixed_test_set[fixed_test_set['TRUE_SENTIMENT'] == 'Positive']),
                        len(fixed_test_set[fixed_test_set['TRUE_SENTIMENT'] == 'Neutral']))

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


