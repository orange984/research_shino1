#!/bin/bash
EPOCHS=(100)
SIZES=(100)


for INDEX in 0; do
  python train_n20.py --problem_size ${SIZES[$INDEX]}\
                      --pomo_size ${SIZES[$INDEX]}\
                      --path "./result/saved_tsp""${SIZES[$INDEX]}""_model"\
                      --epoch ${EPOCHS[$INDEX]}\
                      --NORM_MODE\
                      --train_episodes 10000\
                      --train_batch_size 20\
                      --desc "test__tsp_n""${SIZES[$INDEX]}"
done
