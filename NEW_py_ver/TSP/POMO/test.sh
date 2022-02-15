#!/bin/bash
EPOCHS=(510)
SIZES=(20)
AUGS=(8)

for INDEX in 0; do
for AUG in "${AUGS[@]}"; do
  python test_n20.py  --DEBUG_MODE\
                      --problem_size ${SIZES[$INDEX]}\
                      --pomo_size 1\
                      --path "./result/saved_tsp""${SIZES[$INDEX]}""_model"\
                      --epoch ${EPOCHS[$INDEX]}\
                      --NORM_MODE\
                      --TEST_MODE\
                      --test_set "../TSProblem/testset_n""${SIZES[$INDEX]}"".npy"\
                      --test_episodes 10000\
                      --test_batch_size 20\
                      #--augmentation_enable\
                      --aug_factor $AUG\
                      --aug_batch_size 20\
                      --desc "test__tsp_n""${SIZES[$INDEX]}"
done
done
