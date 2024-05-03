#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import json
data_path = "../../data/out"

with open(f"{data_path}/sampled_entries.json") as file:
    sampled = json.load(file)

with open(f"{data_path}/translated_entries.json") as file:
    transd = json.load(file)

for i, band in enumerate(sampled):
    for j, entry in enumerate(band["entries"]):
        sampled[i]["entries"][j].append(transd[i]["entries"][j])

with open(f"{data_path}/merged_entries.json", "w+") as file:
    json.dump(sampled, file)