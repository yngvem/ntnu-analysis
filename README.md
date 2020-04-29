# Code for autodelineation experiments on MRI data

Start by running `setup.sh` to download the singularity container
Then, submit slurm jobs like this:

```bash
sbatch slurm.sh json/dice/dwi.json dwi_dice 200
```

Which will load the setup from the `json/dice/dwi.json` file, train for 200 epochs
and store the results in the folder `$HOME/logs/ntnu/dwi_dice/`.