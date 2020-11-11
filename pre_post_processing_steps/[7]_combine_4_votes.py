#!/usr/bin/env python
# coding: utf-8
#### compute agreement between the new batch and previous batch ( when some batches are submitted for reannotating, this code
#### will combine hits with 3 and 4 votes, remove the most divers answer out of 4 answers and create an output like all other outputs
#### with three answers per hit


import pandas as pd
new_batch_file = 'data/mturk_out_aug19_part2_v2.csv'
pre_batch_file = 'data/mturk_out_aug19_part2.csv' 

new_batch = pd.read_csv(new_batch_file)
pre_batch = pd.read_csv(pre_batch_file)
new_titles = new_batch['Input.title']
pre_titles = pre_batch['Input.title']


new_df_appended = pd.DataFrame()
new_df_notappended = pd.DataFrame()
for title in new_titles:
    entity = list(new_batch[new_batch['Input.title'] == title]['Input.entity'])[0]
    hitid = list(pre_batch[pre_batch['Input.title'] == title]['HITId'])[0]
    pre_df = pre_batch[pre_batch['Input.title'] == title]
    pre_df = pre_df [ pre_df['Input.entity'] == entity]
    new_df_appended = new_df_appended.append(pre_df)
    batch = new_batch[new_batch['Input.title'] == title]
    batch['HITId'] = hitid

    new_df_appended = new_df_appended.append(batch)
    new_df_notappended= new_df_notappended.append(pre_df)

print(len(new_batch),len(pre_batch),len(new_df_appended),len(new_df_notappended))
new_df_appended.to_csv('appended_part2.csv')


hitids = new_df_appended['HITId'].unique()
arr_map = {'Negative': -2, 'Slightly_Negative':-1 , 'Neutral': 0 , 'Slightly_Positive': 1, 'Positive':2}
sent_arr = []
counter = 0
to_be_removed_indices = []
for hid in hitids:
    hit_arr = []
    df = new_df_appended[new_df_appended['HITId']==hid]
    df1 = df[df['AssignmentStatus']=='Approved']
    if len(df1)>3:
        sentiments = list(df1['Answer.Final_sentiment'])
        if np.nan in sentiments:
            remove_idx = sentiments.index(np.nan)
        else:
            for item in sentiments:
                hit_arr.append(arr_map[item])
            min_var = 1000
            for index in range(len(hit_arr)-1,-1,-1):
                new_arr = [hit_arr[idx] for idx in range(len(hit_arr))  if idx != index  ]
                new_var = np.var(new_arr)
                if new_var < min_var:
                    min_var = new_var
                    remove_idx = index
        to_be_removed_indices.append(counter+remove_idx)
        
    counter += 4
        
cleaned_df = new_df_appended.drop(new_df_appended.index[to_be_removed_indices],0)
cleaned_df.to_csv('mturk_out_combined_versions_part2.csv')