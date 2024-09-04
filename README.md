
## Datasets
We test our algorithms on the Amazon review dataset. five cross-domain recommendation tasks are built upon these four datasets: Movie->Music (Task1), Books->Music (Task2), Books->Movies (Task3), Movies->Toys (Task4) and Books->Toys (Task5).
you can use the following link: ![Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/links.html).

## Run
First, reviews are processed by the following command,
```
python reviews_processing --tgt_category Toys_and_Games --src_category Movies_and_TV
```

Then, Train and test datasets are processed by the following command.
```
python data_processing.py --src_category Toys_and_Games --tgt_category Movies_and_TV
```

Finally, train model is by the following command.
```
python main.py --transfer_types='group' --feature_types 'category' 'item' 'aspect' 'avg'  --src_category Toys_and_Games --tgt_category Movies_and_TV
```


