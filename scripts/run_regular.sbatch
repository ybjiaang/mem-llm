#!/bin/bash
#
#SBATCH --mail-user=yiboj@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/yiboj/slurm/out/%j.%N.stdout
#SBATCH --error=/home/yiboj/slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/yiboj/mem-llm
#SBATCH --partition=general
#SBATCH -c 24
#SBATCH --mem 320G

source activate torch-llm

python -u main.py config_file=$CONFIG exp_id=$EXP_ID
