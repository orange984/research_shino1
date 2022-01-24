#!/bin/bash
EPOCHS=(510 1000 2000)
SIZES=(20 50 100)
AUGS=(8 10 12 16)

for INDEX in 0 1 2; do
for AUG in "${AUGS[@]}"; do
  python test_n20.py  --problem_size ${SIZES[$INDEX]}\
                      --pomo_size ${SIZES[$INDEX]}\
                      --path "./result/saved_tsp""${SIZES[$INDEX]}""_model"\
                      --epoch ${EPOCHS[$INDEX]}\
                      --test_episodes 100000\
                      --test_batch_size 10000\
                      --augmentation_enable\
                      --aug_factor $AUG\
                      --aug_batch_size 1000\
                      --desc "test__tsp_n""${SIZES[$INDEX]}"
done
done
