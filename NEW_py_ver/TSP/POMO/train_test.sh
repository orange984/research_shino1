#!/bin/bash
EPOCHS=(510)
SIZES=(20)


for INDEX in 0; do
  python train_n20.py --DEBUG_MODE\
                      --problem_size ${SIZES[$INDEX]}\
                      --pomo_size ${SIZES[$INDEX]}\
                      --path "./result/saved_tsp""${SIZES[$INDEX]}""_model"\
                      --epoch ${EPOCHS[$INDEX]}\
                      --train_episodes 10000\
                      --train_batch_size 20\
                      --desc "test__tsp_n""${SIZES[$INDEX]}"
done
