#!/bin/bash

src=$1
tgt=$2
gpu=$3

num_workers=12
bz=1024
mode=train


for seed in 42 43 44;do
  for user_proportions in 0.2 0.5 0.8;do
  python main.py --transfer_types='group'  --batch_size=$bz --mode=$mode --feature_types 'category' 'brand' 'aspect' --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
  if [ $user_proportions = 0.2 ];then
  python main.py --transfer_types='personal' --batch_size=$bz --mode=$mode --feature_types 'category' 'brand' 'aspect' --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
  python main.py --transfer_types='group'  --batch_size=$bz --feature_types 'category'  --mode=$mode --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
  python main.py --transfer_types='group'  --batch_size=$bz --feature_types 'category' 'brand'  --mode=$mode --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
  fi
  done
done

