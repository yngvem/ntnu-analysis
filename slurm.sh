#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=my_job_name   # sensible name for the job
#SBATCH --mem=16G                 # Default memory per CPU is 3GB.
#SBATCH --partition=gpu # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --gres=gpu:1
#SBATCH --mail-user=yngvemoe@nmbu.no # Email me when job is done.
#SBATCH --mail-type=ALL
#SBATCH --output=outputs/unet-%A.out
#SBATCH --error=outputs/unet-%A.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

## Code
# If data files aren't copied, do so
if [ ! -d $TMPDIR/ntnu_delin ]; then
    mkdir $TMPDIR/ntnu_delin
    cp -r $HOME/datasets/ntnu $TMPDIR/ntnu_delin/
fi

# Hack to ensure that the GPUs work
nvidia-modprobe -u -c=0

# Run experiment
singularity exec --nv deoxys.sif python unet.py $1 $2 --epochs $3