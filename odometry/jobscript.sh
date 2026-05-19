#!/bin/sh
### General options
### –- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J odom_main
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- set walltime limit: hh:mm
#BSUB -W 3:00
# request 20GB of system-memory
#BSUB -R "rusage[mem=20GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o cpu_%J.out
#BSUB -e cpu_%J.err
# -- end of LSF options --

set -e

cd /work3/s250471/special_course_radar_imaging

echo "Working directory: $(pwd)"
echo "Starting odometry main at $(date)"

# Use local project venv if available.
if [ -f .venv/bin/activate ]; then
    . .venv/bin/activate
fi

python --version

python odometry/main_odometry.py

echo "Finished odometry main at $(date)"