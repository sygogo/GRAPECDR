#!/bin/bash

src=$1
tgt=$2

for user_proportions in 0.5 0.3;do
  python data_processing.py --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
done


python reviews_processing.py --llm_url=http://192.168.60.175:8000 --src_category=Movies_and_TV --tgt_category=CDs_and_Vinyl --test_proportions=0.2
python reviews_processing.py --llm_url=http://192.168.60.175:8001 --src_category=Movies_and_TV --tgt_category=CDs_and_Vinyl --test_proportions=0.5
python reviews_processing.py --llm_url=http://192.168.1.133:8000 --src_category=Books --tgt_category=CDs_and_Vinyl --test_proportions=0.2
python reviews_processing.py --llm_url=http://192.168.1.133:8001 --src_category=Books --tgt_category=CDs_and_Vinyl --test_proportions=0.5
python reviews_processing.py --llm_url=http://192.168.0.237:8000 --src_category=Books --tgt_category=Movies_and_TV --test_proportions=0.2
python reviews_processing.py --llm_url=http://192.168.0.237:8001 --src_category=Books --tgt_category=Movies_and_TV --test_proportions=0.5