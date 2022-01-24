#!/usr/sh

for var in 20 50 100; do
  python test_n20.py  --problem_size $var\
                      --pomo_size $var\
                      --test_episodes 100000\
                      --test_batch_size 10000\
                      --augmentation_enable True\
                      --aug_factor 8\
                      --aug_batch_size 1000
done
