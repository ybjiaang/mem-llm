#!/bin/bash

nb_concepts=(5 6 7 8)
dims=(16 32 64 128)
exp_ids=(0 1 2 3 4)
lrs=(0.01 0.001)
# lengths=(8 16 24 32 40 48)
lengths=(64 128)

for NB_CONCEPTS in "${nb_concepts[@]}"
do

for DIM in "${dims[@]}"
do

for EXP_ID in "${exp_ids[@]}"
do

for LR in "${lrs[@]}"
do

for LEN in "${lengths[@]}"
do

CONFIG=config/length/config_n${NB_CONCEPTS}_d${DIM}_lr${LR}_l${LEN}.yaml

echo $CONFIG, $EXP_ID
export CONFIG EXP_ID

sbatch --job-name=nb${NB_CONCEPTS} scripts/run_regular.sbatch

sleep 1

done
done
done
done
done
