#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
python main.py --cfg configs/condense_single_epinions.yaml --repeat 10
