#!/bin/bash

nb_concepts=(5 6 7 8)
exp_ids=(0 1 2 3 4 5 6 7 8 9)

for NB_CONCEPTS in "${nb_concepts[@]}"
do

for EXP_ID in "${exp_ids[@]}"
do


CONFIG=config/no_wv_low_dim/config_n${NB_CONCEPTS}.yaml

echo $CONFIG, $EXP_ID
export CONFIG EXP_ID

sbatch --job-name=nb${NB_CONCEPTS} scripts/run_regular.sbatch

sleep 1

done
done
