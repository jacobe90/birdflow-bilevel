import os
import random

IDX = int(os.getenv("SLURM_ARRAY_TASK_ID"))

random_bit = random.randint(0, 1)
if random_bit == 0:
    print(f"hello {IDX}")
else:
    print("Oh no!") # error