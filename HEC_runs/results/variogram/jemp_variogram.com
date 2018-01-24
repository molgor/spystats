## Specifies to use the BASH as interpreter.
#$ -S /bin/bash
## Defines a Serial Queue will run in "single-core" jobs.
#$ -q serial

## Sets a name for the job for easy identification. This name will also be used for creating the output files.
#$ -N emp_variogram 

## Send email notifications
#$ -m e
#$ -M j.escamillamolgora@lancaster.ac.uk


## Initializes the appropriate environment.
source /etc/profile
source $HOME/.profile

echo "Job running on compute node" `uname -n` 


## Test small region
## python /home/hpc/28/escamill/spystats/HEC_runs/variogram_fia.py /home/hpc/28/escamill/csv_data/idiv/plotsClimateData_11092017.csv -90 -85 30 35

python /home/hpc/28/escamill/spystats/HEC_runs/variogram_fia.py /home/hpc/28/escamill/csv_data/idiv/plotsClimateData_11092017.csv -130 -60 25 50 

#### qsub -l node_type=10Geth128G -l env=cento7 -l h_vmem=126G -M j.escamillamolgora@lancaster.ac.uk jemp_variogram.com

#### qsub -q night -l env=cento7 -l h_vmem=126G -M j.escamillamolgora@lancaster.ac.uk jemp_variogram.com

 
