#!/bin/bash

# basin of attraction
nb_concepts=(5 6 7 8)
dims=(16 32 64 128 256 512)
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

CONFIG=config/dim_freeze/config_n${NB_CONCEPTS}_d${DIM}.yaml

echo $CONFIG, $EXP_ID
export CONFIG EXP_ID

sbatch --job-name=nb${NB_CONCEPTS} scripts/run_regular.sbatch

sleep 1

done
done
done
