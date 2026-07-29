#!/bin/bash
#SBATCH -p shakhnovich,sapphire         # Partitions to submit to
#SBATCH -N 1                          # One node
#SBATCH -n 1                          # One task per job
#SBATCH -c 1                          # One core per task
#SBATCH --mem=3000                    # Memory in MB per task
#SBATCH -t 10:00:00                   # Walltime
#SBATCH -o optim_outfile.out    
#SBATCH -e optim_errfile.err    # Stderr per task (%A=jobID, %a=arrayID)
#SBATCH --array=0-11                  # 3 hf × 4 tau × 3 N configs = 36 tasks

# Activate environment
source activate lantern

# Define parameter arrays
HF_VALUES=(30)
TAU_VALUES=(10 20 30 40)
N_VALUES=(1 2 3)

# Dimensions
NUM_HF=${#HF_VALUES[@]}
NUM_TAU=${#TAU_VALUES[@]}
NUM_N=${#N_VALUES[@]}

# Compute indices based on SLURM_ARRAY_TASK_ID
hf_index=$(( SLURM_ARRAY_TASK_ID / (NUM_TAU * NUM_N) ))
rem=$(( SLURM_ARRAY_TASK_ID % (NUM_TAU * NUM_N) ))
tau_index=$(( rem / NUM_N ))
N_index=$(( rem % NUM_N ))

# Extract values for this specific array task
hf="${HF_VALUES[$hf_index]}"
tau="${TAU_VALUES[$tau_index]}"
N="${N_VALUES[$N_index]}"

# Set C_inj_init, t_inj_init, and lr_C based on N and tau
if [ "$N" -eq 1 ]; then
    lr_C=1
    if [ "$tau" -eq 10 ]; then C_init="3941.6814"; T_init="0.0"; fi
    if [ "$tau" -eq 20 ]; then C_init="119.14573"; T_init="0.0"; fi
    if [ "$tau" -eq 30 ]; then C_init="37.065907"; T_init="0.0"; fi
    if [ "$tau" -eq 40 ]; then C_init="20.684076"; T_init="0.0"; fi

elif [ "$N" -eq 2 ]; then
    lr_C=1
    if [ "$tau" -eq 10 ]; then C_init="207.23526 64.861336"; T_init="0.0 75.91039"; fi
    if [ "$tau" -eq 20 ]; then C_init="34.081005 10.942306"; T_init="0.0 80.972244"; fi
    if [ "$tau" -eq 30 ]; then C_init="18.197401 5.4910116"; T_init="0.0 85.50722"; fi
    if [ "$tau" -eq 40 ]; then C_init="13.108559 3.5591366"; T_init="0.0 89.6414"; fi

elif [ "$N" -eq 3 ]; then
    lr_C=1
    if [ "$tau" -eq 10 ]; then C_init="98.58391 20.342106 20.621147"; T_init="0.0 57.207687 98.5856"; fi
    if [ "$tau" -eq 20 ]; then C_init="26.153332 5.668279 5.725831"; T_init="0.0 65.025894 102.456345"; fi
    if [ "$tau" -eq 30 ]; then C_init="15.940841 3.17092 3.194743"; T_init="0.0 71.53222 105.68527"; fi
    if [ "$tau" -eq 40 ]; then C_init="12.141661 2.1264074 2.1385903"; T_init="0.0 77.21593 108.503975"; fi
fi

echo "Task $SLURM_ARRAY_TASK_ID starting: hf=$hf, tau=$tau, N=$N"
echo "Using parameters: C_init=[$C_init], T_init=[$T_init], lr_C=$lr_C"

# Run Python script
python3 script_vaccine_optim.py \
    --C_inj_init $C_init \
    --t_inj_init $T_init \
    --target_hf $hf \
    --num_iterations 50 \
    --lr_C $lr_C \
    --tau $tau \
    --lr_t 1 \
    --alpha_extinction 10 \
    --clipbound 1.33 \
    --output_dir results_1_1_alpha10_clipbound1.33

echo "Task $SLURM_ARRAY_TASK_ID done."