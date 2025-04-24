import os
import re

error_ids = []
for i in range(3):
    root = f"/work/pi_drsheldon_umass_edu/birdflow_modeling/jacob_independent_study/birdflow-bilevel/logs/w2-grid-search/run{i}-logs"
    for file in os.listdir(root):
        with open(os.path.join(root, file), 'r') as f:
            contents = f.read()
            msg3 = "CUDA_ERROR_ILLEGAL_ADDRESS"
            msg4 = "CUDA_ERROR_ILLEGAL_INSTRUCTION"
            if  msg3 in contents or msg4 in contents: # we have an error,
                match = re.search(r'_(\d+)\.out', file)
                error_ids.append(int(match.group(1)))
print(error_ids)