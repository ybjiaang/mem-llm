#!/bin/bash

# basin of attraction
nb_concepts=(5 6 7 8)
# nb_concepts=(8)
dims=(16 32 64 128 256 512)
lrs=(0.01 0.001)
# dims=(64)
exp_ids=(0 1 2 3 4)
# nb_concepts=(5)
# dims=(16)
# exp_ids=(0)

for NB_CONCEPTS in "${nb_concepts[@]}"
do

for DIM in "${dims[@]}"
do

for EXP_ID in "${exp_ids[@]}"
do

for LR in "${lrs[@]}"
do

CONFIG=config/dim_no_freeze_sweep/config_n${NB_CONCEPTS}_d${DIM}_lr${LR}.yaml

echo $CONFIG, $EXP_ID
export CONFIG EXP_ID

sbatch --job-name=nb${NB_CONCEPTS} scripts/run_regular.sbatch

sleep 1

done
done
done
done
