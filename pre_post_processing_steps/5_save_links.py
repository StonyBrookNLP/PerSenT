import json
import os
import requests
import pickle
import pandas as pd
# 
# path = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_resource/EMNLP_Data_Junting/article_url_cmplt.json'
# with open(path) as article_url_file:
#     article_url_map = json.load(article_url_file)
# 
# 
# headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
# def add_url_to_archie(filename_archive_url,path,URL):   
#     i = 0
#     err = 0
#     vld = 0
#     for filename in os.listdir(path):
#         i += 1
# 
#         if i%100 ==0:
#             print('i is %d'%i)
#         if filename in filename_archive_url:
#             vld+= 1
#             continue
#         added_url = article_url_map[filename] 
#         try:
#             r2 = requests.get(url = URL+'/save/'+added_url,headers=headers)
#         except Exception as e:
#             print('error occured: %s'%str(e))
#             continue
#         if r2.status_code == 502:
#             for _ in range(3):
#                 print('attempt 10 times to get something else than 502')
#                 r2 = requests.get(url = URL+'/save/'+added_url,headers=headers)
#                 if r2.status_code == 200:
#                     print('succeed')
#                     break
#         if r2.status_code == 200:
#                 new_url = r2.headers['Content-Location']
#                 
# #             if added_url in new_url:
#                 rsp = requests.head(url = URL+new_url,headers=headers)
#                 if rsp.status_code == 200:
#                     filename_archive_url[filename] = URL+new_url
#                     vld+= 1
#                 else:
#                     err += 1
#                     print('error on file  %s with url %s with error code: %d (while saving)'%(filename,added_url,rsp.status_code))
#                     continue                    
# #             else:
# #                 print('added url %s'%(added_url))
# #                 print('new url %s'%(new_url))
# #                 err += 1
# #                 print('error on file %s with url %s with error code: %d ( new url not same as requested url)'
# #                       %(filename,added_url,r2.status_code))
# #                 continue
# 
#         else:
#             err += 1
#             print('error on file %s with url %s with error code: %d '%(filename,added_url,r2.status_code))
#             continue
# 
#     print('total: %d, valid: %d, error: %d, len map: %d'%(i,vld,err,len(filename_archive_url)) )
#     return(filename_archive_url) 
# 
#     
# URL = 'http://web.archive.org'
# filename_archive_url = {}
# 
# path = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp_paragraph_seperated_Aug19_part2/'
# filename_archive_url = add_url_to_archie(filename_archive_url,path,URL)
# 
# 
# ### check all urls in the archive to be valid
# import requests
# i = 0
# should_be_deleted = []
# for key in filename_archive_url:
#     i += 1
#     if i % 100 == 0:
#         print('i is %d'%i)
#     url = filename_archive_url[key]
# #     if 'err' in url:
# #         print('error in address %s, key is %s'%(url,key))
#     rr = requests.head(url = url,headers=headers)
#     if rr.status_code != 200:
#         print(rr.status_code)
#         print('error in file %s with url %s'%(key,article_url_map[key]))
#         should_be_deleted.append(key)
# print('documents which should be deleted: \n', should_be_deleted)
# 
# pickle.dump(filename_archive_url,open('file_url_Aug19_part3','wb'))
# 
# 
# 
# ### check the folders to see how many news artciles were not retrieved with status code
# filename_ids_notadded = []
# URL = 'http://web.archive.org'
# path = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/dataAnalysis/emnlp18_data/used_documents_in_MTurk_emnlp_paragraph_seperated_Aug19_part2/'
# 
# for path in [path]:
#     for filename in os.listdir(path):
#         if not filename.startswith('_'):
#             if filename not in filename_archive_url:
#                 filename_ids_notadded.append(filename)
#                 added_url = article_url_map[filename] 
#                 r2 = requests.get(url = URL+'/save/'+added_url)
#                 status = r2.status_code
#                 print('error on file %s with url %s with status %d'%(filename,added_url,status))
#         else:
#             filename_ids_notadded.append(filename[1:])
# 
# pickle.dump(filename_ids_notadded,open('file_not_added_Aug19_part3','wb'))
# 

def map_title_doc():

	#### find the titles for specific doc ids and map title to doc id for both removed docs and kept docs
	titles = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_resource/EMNLP_Data_Junting/titles/title_selected_sources_ids_with_targets_'
	df_titles = pd.DataFrame([])
	for name in ['train','test','val']:
		df_file = pd.read_csv(open(titles+name+'.csv'),sep='\t')
	#     print(len(df_file))
		df_titles = df_titles.append(df_file)
		print(len(df_titles))
	print('--------- Dataframe of titles created ---------') 

	docs = '/Users/mohaddeseh/Documents/EntitySentimentAnalyzer-master/data_resource/EMNLP_Data_Junting/content_df_'
	df_docs = pd.DataFrame([])
	for name in ['train','test','val']:
		df_file = pd.read_csv(open(docs+name+'_filtered.csv'),sep='\t')
	#     print(len(df_file))
		df_docs = df_docs.append(df_file)
		print(len(df_docs))
	print('--------- Dataframe of documents created ---------')  

	to_be_deleted_title = {}
	to_be_deleted_doc = {}
	i = 0
	for item in list(df_titles['docid']):
		if str(item) in filename_ids_notadded:
			to_be_deleted_title[item] = df_titles[df_titles['docid']==item]['title'].values[0]
			to_be_deleted_doc[item] = df_docs[df_docs['docid']==item]['content'].values[0]
		i += 1

	to_be_kept_title = {}
	to_be_kept_doc = {}
	i = 0 
	for item in list(df_titles['docid']):
		if str(item) in filename_archive_url:
			to_be_kept_title[item] = df_titles[df_titles['docid']==item]['title'].values[0]
			to_be_kept_doc[item] = df_docs[df_docs['docid']==item]['content'].values[0]
		i += 1
	print('--------- Deleting and keeping lists of title and documents created ---------')   
	print('--------- Statistics about the dataset ---------') 
	print('length of the list of titles to be deleted: %d'%len(to_be_deleted_title))
	print('length of the list of titles to be kept: %d'%len(to_be_kept_title))
	print('length of the set of titles to be kept: %d'%len(set(to_be_kept_title.values())))
	print('length of the list of docs to be deleted: %d'%len(to_be_deleted_doc))
	print('length of the list of docs to be kept: %d'%len(to_be_kept_doc))
	print('length of the set of docs to be kept: %d'%len(set(to_be_kept_doc.values())))

	### If we have repeated docs, remove one of them by adding it to the to_be_deleted list 
	print('--------- List of the doc ids which has the same content ---------') 
	repeated_docs = []
	key_list = list(to_be_kept_doc)

	for i in  range(len(key_list)):
		key1 = key_list[i]
		value1 = to_be_kept_doc[key1]
		for j in range(i+1,len(key_list)):
			key2 = key_list[j] 
			value2 = to_be_kept_doc[key2]
			if key1 != key2 and value1 == value2:
				print(key1,key2)
				repeated_docs.append(key1)
				to_be_deleted_doc[key1] = value1
				to_be_deleted_title[key1] = to_be_kept_title[key1]
	#             print(value1)
	print('--------- delete the repeated docs from the list of kept docs ---------') 
	for key in repeated_docs:   
		del to_be_kept_doc[key]    
		del to_be_kept_title[key] 
	print('length of the list of docs to be kept: %d'%len(to_be_kept_doc))  
	return (to_be_deleted_doc, to_be_deleted_title)
	
	
	
def del_unsused_doc_title(to_be_deleted_title,data_names=['part3'],path = './'):
	### remove document which doesn't have any link in archive in mturk input

	remove_from_dataset = {}
	### collect the ids
	for name in data_names:
		print('-----------------\tAnalyzing %s\t-----------------'%name)
		data_path =  path + 'input_emnlp_PS_Aug19_%s.csv'%(name)
		df = pd.read_csv(data_path)
		remove_set = []
		i = 0
		print(len(df['title']))
		for title in df['title']:
			if title in to_be_deleted_title.values():
				remove_set.append(i)
			i += 1
		remove_from_dataset[name] = remove_set
	print(remove_from_dataset)
	
	### remove the collected ids
	for dataset in remove_from_dataset.keys():
		print('working on dataset %s'%dataset)

		data_path =  path + 'input_emnlp_PS_Aug19_%s.csv'%(dataset)
		df = pd.read_csv(data_path)
		print('len df before dropping: %d'%len(df))
		drop_list = remove_from_dataset[dataset]
		print(drop_list)
		df = df.drop(df.index[drop_list])
		print('len df after dropping: %d'%len(df))
		df.to_csv(path + 'input_emnlp_PS_dropped_Aug19_%s.csv'%(dataset),index=False)
	


filename_archive_url = pickle.load(open('data/file_url_Aug19_part3_cop','rb'))
filename_ids_notadded = pickle.load(open('data/file_not_added_Aug19_part3_cop','rb'))
(to_be_deleted_doc, to_be_deleted_title) = map_title_doc()
del_unsused_doc_title(to_be_deleted_title,data_names=['part3'],path = './data/')


