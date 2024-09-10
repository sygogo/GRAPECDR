#!/bin/bash

src=$1
tgt=$2

for user_proportions in 0.5 0.3;do
  python data_processing.py --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
done


