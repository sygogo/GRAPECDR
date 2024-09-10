
## Dataset Preparation
We test our algorithms on the Amazon review dataset. and three cross-domain recommendation tasks are built using the following datasets:
Task 1: Movie → Music 
Task 2: Books → Music
Task 3: Books → Movies

you can use the following link: [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/links.html). Download the three domains: 
CDs and Vinyl, Movies and TV, Books (5-scores/ratings_only/reviews), and then put the data in ./data/meta 


## Run
First, reviews are processed by the following command, process the reviews using the following commands. 
The data/meta directory contains the downloaded datasets (CDs and Vinyl, Movies and TV, Books). The processed data will be stored in data/processed. Be aware that results may vary based on the llm_url value. 
The data/raw directory stores the train/validation/test data processed by  [CATN](https://github.com/AkiraZC/CATN). 
you can use my data in [data](https://drive.google.com/drive/folders/1bezCXI5yK4WtgWxzDHS_Qoaa0wtjrZPG?usp=drive_link).

```
python data_processing.py --raw_data_path data/raw --processed_data_path data/processed --meta_data_path data/meta --tgt_category CDs_and_Vinyl --src_category Books --user_proportions 0.2
python reviews_processing.py --llm_url= --src_category Books --tgt_category CDs_and_Vinyl --test_proportions 0.2
```

Once the data is processed, train and test the datasets using the following command. Adjust the parameters based on your setup:
```
python main.py --transfer_types='group' --batch_size=$bz --mode=$mode --feature_types 'category' 'brand' 'aspect' --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
```
Make sure to replace placeholders like $bz, $mode, $gpu, $src, and $tgt with your specific configurations.



