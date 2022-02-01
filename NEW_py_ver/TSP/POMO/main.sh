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
                      --test_episodes 10000\
                      --TEST_MODE\
                      --test_set "../TSProblem/testset_n""${SIZES[$INDEX]}"".npy"\
                      --test_batch_size 20\
                      --augmentation_enable\
                      --aug_factor $AUG\
                      --aug_batch_size 20\
                      --desc "test__tsp_n""${SIZES[$INDEX]}"
done
done
