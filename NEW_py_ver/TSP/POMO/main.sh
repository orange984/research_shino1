#!/bin/bash
EPOCHS=(2000)
SIZES=(100)
AUGS=(8 12 16)

for INDEX in 0; do
for AUG in "${AUGS[@]}"; do
  python test_n20.py  --problem_size 250\
                      --pomo_size 250\
                      --path "./result/saved_tsp""${SIZES[$INDEX]}""_model"\
                      --epoch ${EPOCHS[$INDEX]}\
                      --test_episodes 10000\
                      --test_batch_size 20\
                      --augmentation_enable\
                      --aug_factor $AUG\
                      --aug_batch_size 20\
                      --desc "test__tsp_n""${SIZES[$INDEX]}"
done
done
