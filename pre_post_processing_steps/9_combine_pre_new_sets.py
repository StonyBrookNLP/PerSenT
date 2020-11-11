import pandas as pd
sets_name = ['train','dev','random_test','fixed_test']
pre_set_prefix = 'alldata_3Dec_7Dec_PS_reindex_enclosed_masked'
new_set_prefix = 'alldata_aug19_enclosed_masked'
pre_sets = [] # train, dev, random_test, fixed_test
new_sets = []
concat_sets= []
shuffle_sets = []

column_names=['DOCUMENT_INDEX','TITLE','TARGET_ENTITY','DOCUMENT','MASKED_DOCUMENT','TRUE_SENTIMENT','Paragraph0','Paragraph1','Paragraph2','Paragraph3','Paragraph4','Paragraph5','Paragraph6','Paragraph7','Paragraph8','Paragraph9','Paragraph10','Paragraph11','Paragraph12','Paragraph13','Paragraph14','Paragraph15']

for set_name in sets_name:
	 pre_set = pd.read_csv('./pre_set/%s_%s_v3.csv'%(str(pre_set_prefix),str(set_name)))
	 pre_sets.append(pre_set)
	 
	 new_set = pd.read_csv('./new_set/%s_%s.csv'%(str(new_set_prefix),str(set_name)),encoding = "ISO-8859-1")
	 new_set['DOCUMENT_INDEX'] = new_set['DOCUMENT_INDEX']+ 3000
	 new_sets.append(new_set)
	 concat_set = pd.concat([pre_set, new_set[['TARGET_ENTITY','DOCUMENT_INDEX','TITLE','DOCUMENT','TRUE_SENTIMENT','Paragraph0','Paragraph1','Paragraph2','Paragraph3','Paragraph4','Paragraph5','Paragraph6','Paragraph7','Paragraph8','Paragraph9','Paragraph10','Paragraph11','Paragraph12','Paragraph13','Paragraph14','Paragraph15','MASKED_DOCUMENT']]],sort=True,ignore_index=True)
# 	 concat_set = pd.concat([pre_set, new_set],names=['DOCUMENT_INDEX','TITLE','TARGET_ENTITY','DOCUMENT','MASKED_DOCUMENT','TRUE_SENTIMENT','Paragraph0','Paragraph1','Paragraph2','Paragraph3','Paragraph4','Paragraph5','Paragraph6','Paragraph7','Paragraph8','Paragraph9','Paragraph10','Paragraph11','Paragraph12','Paragraph13','Paragraph14','Paragraph15'],sort=False,ignore_index=True)
	 concat_set = concat_set[column_names]
	 shuffle_set = concat_set.sample(frac=1)
	 concat_sets.append(concat_set)
	 shuffle_sets.append(shuffle_set)
	 shuffle_set.to_csv('all_data_combined_shuffled_3Dec_7Dec_aug19_%s.csv'%str(set_name),index=False)
	