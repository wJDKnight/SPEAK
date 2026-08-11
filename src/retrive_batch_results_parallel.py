# this script is used to retrieve the results of the parallel runs
from openai import OpenAI
import pandas as pd
import os
import re
import time
from .utils import check_batch_status, load_config

# get the path of config file from the command line
import sys
config_path = sys.argv[1]
config = load_config(config_path)


config.data_name = sys.argv[2]
config.refresh_paths()

config.replicate = sys.argv[3]

timestamp = sys.argv[4]
client = OpenAI()

# read the batch_id_file
with open(f"outs/{config.data_name}{config.replicate}_{timestamp}.txt", 'r') as f:
    batch_ids = f.readlines()

for batch_id in batch_ids:
    batch_id = batch_id.strip()
    file_response = client.files.content(client.batches.retrieve(batch_id).output_file_id)
    # get the number after the _batch
    number = re.search(r'_batch(\d+)', client.batches.retrieve(batch_id).metadata["description"]).group(1)
    save_name = f"response_{config.data_name}_{number}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.txt"
    output_file_name = f"{config.output_path}/{save_name}"
    # Open the file in write mode and save the string
    with open(output_file_name, 'w') as file:
        file.write(file_response.text)  
    print(f"The response has been saved to {output_file_name}")
