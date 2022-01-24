#!/bin/bash
EPOCHS=(2000)
SIZES=(100)
AUGS=(16 8)

for INDEX in 0; do
for AUG in "${AUGS[@]}"; do
  python test_n20.py  --problem_size ${SIZES[$INDEX]}\
                      --pomo_size ${SIZES[$INDEX]}\
                      --path "./result/saved_tsp""${SIZES[$INDEX]}""_model"\
                      --epoch ${EPOCHS[$INDEX]}\
                      --test_episodes 200\
                      --test_batch_size 10\
                      --augmentation_enable\
                      --aug_factor $AUG\
                      --aug_batch_size 10\
                      --desc "test__tsp_n""${SIZES[$INDEX]}"
done
done
