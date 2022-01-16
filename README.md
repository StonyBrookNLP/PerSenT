## What is PerSenT?
### Person SenTiment, a challenge dataset for author's sentiment prediction in news domain.


You can find our paper [Author's sentiment prediction](https://arxiv.org/abs/2011.06128) 

Mohaddeseh Bastan, Mahnaz Koupaee, Youngseo Son, Richard Sicoli, Niranjan Balasubramanian. COLING2020

We introduce PerSenT, a crowd-sourced dataset that captures the sentiment of an author towards the main entity in a news article. This dataset contains annotation for 5.3k documents and 38k paragraphs covering 3.2k unique entities.

### Example
In the following example we see a 4-paragraph document about an entity (Donald Trump). Each paragraph is labeled separately and finally the author's sentiment towards the whole document is mentioned in the last row.


<a href="https://github.com/MHDBST/PerSenT/blob/main/example2.png?raw=true"><img src="https://github.com/MHDBST/PerSenT/blob/main/example2.png?raw=true" alt="Image of PerSenT stats"/></a>


### Dataset Statistics
To split the dataset, we separated the entities into 4 mutually exclusive sets. Due to the nature of news collections, some entities tend to dominate the collection. In our collection,there were four entities which were the main entity in nearly 800 articles.  To avoid these entities from dominating the train or test splits, we moved them to a separate test collection. We split the remaining into a training, dev, and test sets at random. Thus our collection includes one standard test set consisting of articles drawn at random (Test Standard), while the other is a test set which contains multiple articles about a small number of popular entities (Test Frequent).  
<a href="https://github.com/MHDBST/PerSenT/blob/main/data_stats.png?raw=true"><img src="https://github.com/MHDBST/PerSenT/blob/main/data_stats.png?raw=true" alt="Image of PerSenT stats" /></a>

### Download the data
You can download the data set URLs from [here](https://github.com/MHDBST/PerSenT/blob/main/train_dev_test_URLs.pkl)

The processed version of the dataset which contains used paragraphs, document-level, and paragraph-level labels can be download separately as [train](https://github.com/MHDBST/PerSenT/blob/main/train.csv), [dev](https://github.com/MHDBST/PerSenT/blob/main/dev.csv), [random test](https://github.com/MHDBST/PerSenT/blob/main/random_test.csv), and [fixed test](https://github.com/MHDBST/PerSenT/blob/main/fixed_test.csv).

To recreat the results from the paper you can follow the instructions in the readme file from the [source code](https://github.com/StonyBrookNLP/PerSenT/tree/main/pre_post_processing_steps).

### Liked us? Cite us!

 Please use the following bibtex entry:

   ```
@inproceedings{bastan2020authors,
      title={Author's Sentiment Prediction}, 
      author={Mohaddeseh Bastan and Mahnaz Koupaee and Youngseo Son and Richard Sicoli and Niranjan Balasubramanian},
      year={2020},
      eprint={2011.06128},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
   ```
   
   
   

