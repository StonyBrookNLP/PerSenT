# create input file for mechanical turk, it has three types of columns, entity, title, content. content itself may be
# at most 15 paragraphs and at least 5 paragraphs. we disregard the rest. Finally all entities are highlighted not from 
# the corefrenced document, from the main documents.
import os
import re
source = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/dataAnalysis/emnlp18_data/'
import csv
import spacy
import neuralcoref



docStat= {}
# subdir = 'crowdsource/'
# subdir = 'KBP/'
# subdir = 'emnlp/'
# subdir = 'emnlp_paragraph_seperated/'
# subdir = 'emnlp_paragraph_seperated_batch3/'
subdir = 'emnlp_paragraph_seperated_Aug19_part2/'

# read all files in directory (the reports)

# csvfile = open('input_KBP.csv', 'wb')
# csvfile = open('./input_emnlp_PS.csv', 'wb')
csvfile = open('./input_emnlp_PS_Aug19_part3.csv', 'w')

writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer.writerow(['entity','title','content'])
fileCounter = 0
coref = spacy.load('en')
neuralcoref.add_to_pipe(coref)
cnt = 0
longdocs = []
shortdocs = []
notFound = []
countNF = 0
for filename in os.listdir(source+subdir):
    fileCounter += 1
    if fileCounter%100 == 0:
        print('processing document: %d'%fileCounter)

    ### if the file is processes, skip it    
    try:
        open(source+'used_documents_in_MTurk_%s'%subdir+filename)
        print('file found %s' %filename)
        continue
    except:
        pass
    
    

    
    try:
        coref_text = open(source+'pairentity_%s'%subdir + filename)
    except Exception as e:
        if str(e).startswith('[Errno 2]'):
           continue    
        print('error darim: %s'%str(e))
#         print('pair entity file not found %s'%filename)
#         print(filename)
        notFound.append(filename)
        countNF += 1
        continue
    entity = coref_text.readline().rstrip().replace(',',' ')
    
    try:
        main_text =  open(source+subdir+ filename)
        text = main_text.read().decode('utf-8').strip().replace(',',' ')
    except:
        main_text =  open(source+subdir+ filename)
        text = main_text.read().strip().replace(',',' ')
    new_text = ""
    sc = 0
    try:
        doc = coref(text)
    except Exception as e:
        print('there is an error in %s which is %s'%(filename,str(e)))
        continue
    entity_in_coref = False
    corefs = []
    try:
    # find all mentions and coreferences
        for item in doc._.coref_clusters:
        # if the head of cluster is the intended entity then highlight all of the mentions
            if entity in item.main.text or item.main.text in entity:
                
                entity_in_coref = True
                for span in item:
                    corefs.append(span)
        for item in sorted(corefs):         
                pronoun = item.text
                ec = item.start_char
                if ec < sc:
                    continue
                new_text += text[sc:ec] + '<span style="color:red;"> '+pronoun+' </span>' 
                sc = item.end_char
#                 print(new_text)
        new_text += text[sc:]
        
        new_text = new_text.replace(entity,'<span style="color:red;">'+entity+'</span>' )
                
        new_text = new_text.rstrip().replace(',',' ')
        new_text = new_text.replace('|',' ')
        if not entity_in_coref:
            continue
        
    except Exception as e:
        print('the error is %s'%str(e))
        print('coreference not resolved %s' % filename)
        continue
    
    main_lines = new_text.split('\n')
    content = ""
#     if len( main_lines) < 3 :
#         continue
    header = main_lines[0]
    used_text = entity+ '\n'+ header.replace('</span>','').replace('<span style="color:red;">','') + '\n'
    ind=0
    for main_line in main_lines[1:]:
        ### for one paragraph text
#         if main_line.count('<span') < 3 or len(main_line) < 1000:
#             continue
        #### for all documents
        if main_line.find('<span ') == -1:
            continue
        if ind>15:
            break
        main_line = main_line.replace('|',' ')
        i = str(ind)
        #### for one paragraph document
#         SENTIMENT = ''
#         content += '<p>' + main_line + '</p>'
        ##### for multiple paragraph documents
        SENTIMENT = '<div class="btn-group btn-group-justified" data-toggle="buttons" id="Inputs"><label class="btn btn-default"><input id="Negative" name="sentiment_Pargraph_'+i+'" type="radio" value="Negative" />Negative</label> <label class="btn btn-default"> <input id="Slightly_Negative" name="sentiment_Pargraph_'+i+'"  type="radio" value="Slightly_Negative" />Slightly Negative</label> <label class="btn btn-default"> <input id="Neutral" name="sentiment_Pargraph_'+i+'"  type="radio" value="Neutral" />Neutral</label> <label class="btn btn-default"> <input id="Slightly_Positive" name="sentiment_Pargraph_'+i+'"  type="radio" value="Slightly_Positive" />Slightly Positive</label> <label class="btn btn-default"> <input id="Positive" name="sentiment_Pargraph_'+i+'"  type="radio" value="Positive" />Positive</label> </div>'
        content += '<tr class="row_article"><td><p>' + main_line + '</p></td><td>' + SENTIMENT+'</td></tr>'
        
        used_text += main_line.replace('</span>','').replace('<span style="color:red;">','') + '\n'
        
        ind = ind+1
    if ind in docStat:
        
        docStat[ind] = docStat[ind]+1
    else:
        docStat[ind]= 1
    i = str(ind)
    total_sentiment = '<tr class="row_article"><td><p><span style="font-weight: 700;">The whole article\'s view towards <span style="color:red;">'+entity +'</span> is: </span></p></td><td><div class="btn-group btn-group-justified" data-toggle="buttons" id="Inputs"><label class="btn btn-default"><input id="Negative" name="Final_sentiment" type="radio" value="Negative" />Negative</label> <label class="btn btn-default"> <input id="Slightly_Negative" name="Final_sentiment"  type="radio" value="Slightly_Negative" />Slightly Negative</label> <label class="btn btn-default"> <input id="Neutral" name="Final_sentiment"  type="radio" value="Neutral" />Neutral</label> <label class="btn btn-default"> <input id="Slightly_Positive" name="Final_sentiment"  type="radio" value="Slightly_Positive" />Slightly Positive</label> <label class="btn btn-default"> <input id="Positive" name="Final_sentiment"  type="radio" value="Positive" />Positive</label> </div></td></tr>'
    total_sentiment = total_sentiment.replace('|',' ')
#     if ind >3 and ind<8 :
#         fileSize = 'small/'
#         cnt += 1
#         shortdocs.append([entity,header,'<table id="article"  ><tbody>'+content+total_sentiment+'</tbody></table>'])
#     elif ind >7 and ind < 16:
#         fileSize = 'large/'
        
#         longdocs.append([entity,header,'<table id="article"  ><tbody>'+content+total_sentiment+'</tbody></table>'])
# for kbp and crowdsource data
    if ind>3:
# for emnlp data with one paragraph
#     if ind>0:

        row = '<table id="article" ><tbody>'+content+total_sentiment+'</tbody></table>'
        try:
            writer.writerow([entity,header,row])
        except:
            row = unidecode(row) 
            try:
                writer.writerow([entity,header,row])
                
                
            except:
                print('unicode error')
                continue
#     write the paragraphs which are used in a file for future usage
#         datafile = open(source+'used_documents_in_MTurk_merge/'+fileSize+filename,'wb')
        datafile = open(source+'used_documents_in_MTurk_%s'%subdir+filename,'wb')
        datafile.write(used_text.encode('utf-8'))
        datafile.close()
        cnt +=1
    else:
        print('not enough paragraphs have entity mention %s' %filename)
#     if cnt == 1000:
#         break

print("acceptable documents %d"% cnt) 
csvfile.close()