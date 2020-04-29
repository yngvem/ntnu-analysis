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
# Move dataset
mkdir $TMPDIR/ntnu_delin
cp $HOME/datasets/ntnu/mix_compress_downsample.h5 $TMPDIR/ntnu_delin/

# Hack to ensure that the GPUs work
nvidia-modprobe -u -c=0

# Run experiment
singularity exec --nv deoxys.sif python unet.py $1 $2 --epochs $3

# Cleanup
rm $TMPDIR/ntnu_delin/mix_compress_downsample.h5
