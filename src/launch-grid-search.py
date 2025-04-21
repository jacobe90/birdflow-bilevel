# script for launching w2 model grid search

import subprocess
import time
import os
import re

# estimated time for each job
time_per_iter = 60 * 60 * 3 # about 3 hours per iteration

# convert array to comma-separated list of values
def array_to_launchstr(arr):
    return ','.join(str(x) for x in arr)

jobs_to_run = list(range(81))
run_number = 0
while len(jobs_to_run) > 0:
    # launch all jobs in directory
    jobfile_path = '/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/jobfiles/w2-model-grid-search.sh'
    parent_dir_path = '/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/logs/w2-grid-search'
    working_dir_path = os.path.join(parent_dir_path, f'run{run_number}-logs')
    os.mkdir(working_dir_path)
    subprocess.run(["sbatch", f"--array={array_to_launchstr(jobs_to_run)}%10", jobfile_path],
                   cwd=working_dir_path)

    # wait for all array jobs to finish
    # first get job id
    while len(os.listdir(working_dir_path)) == 0:
        time.sleep(5)
    jobfile = os.listdir(working_dir_path)[0]
    job_id = re.search(r"(?<=slurm-)\d+(?=_)", jobfile).group()
    
    # wait for job_id to not be on list of running jobs
    while True:
        result = subprocess.run(["squeue", "--me"], capture_output=True, text=True)
        if job_id not in result.stdout:
            break
        time.sleep(120)
    
    # find jobs which errored out / their ids
    jobfiles = [file for file in os.listdir(working_dir_path) if file.endswith('.out')]
    error_ids = []
    for jobfile in jobfiles:
        with open(os.path.join(working_dir_path, jobfile), "r") as f:
            contents = f.read()
            msg1 = "No visible GPU devices" # error - couldnt find gpu
            msg2 = "DUE TO TIME LIMIT" # error due to timeout
            msg3 = "Oh no!" # test error
            if msg1 in contents or msg2 in contents or msg3 in contents: # we have an error,
                match = re.search(r'_(\d+)\.out', jobfile)
                error_ids.append(int(match.group(1)))
    
    # add these ids to jobs_to_run
    jobs_to_run = error_ids

    run_number += 1