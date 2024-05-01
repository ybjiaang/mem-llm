#!/bin/bash

# basin of attraction
# nb_concepts=(6 7 8 9)
nb_concepts=(5 6 7 8)
# nb_concepts=("config/config_n6.yaml", "config/config_n7.yaml", "config/config_n8.yaml", "config/config_n9.yaml")
exp_ids=(0 1 2 3 4 5 6 7 8 9)
# exp_ids=(0)

for NB_CONCEPTS in "${nb_concepts[@]}"
do

for EXP_ID in "${exp_ids[@]}"
do

CONFIG=config/cluster/config_n${NB_CONCEPTS}.yaml

echo $CONFIG, $EXP_ID
export CONFIG EXP_ID

sbatch --job-name=nb${NB_CONCEPTS} scripts/run_regular.sbatch

sleep 1

done
done
