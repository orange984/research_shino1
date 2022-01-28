#!/bin/bash


for AUG in 32 64 128 ; do
  python test_n20.py  --problem_size 250\
                      --pomo_size $AUG\
                      --path "./result/saved_tsp100_model"\
                      --epoch 2000\
                      --test_episodes 10000\
                      --test_batch_size 100\
                      #--augmentation_enable\
                      --aug_factor $AUG\
                      --aug_batch_size 100\
                      --desc "test__tsp_n100"
done
