#!/usr/bin/python3

import os
import json
import argparse
import pickle

parser = argparse.ArgumentParser(description='Process generate id to path pickle.')
parser.add_argument('base_dir', help='base dir of images')
parser.add_argument('input_json', help='caption json file')
parser.add_argument('output_pickle', help='output pickle')
args = parser.parse_args()

with open(args.input_json) as fd:
    data = json.load(fd)['images']

output = {i['imgid']: os.path.join(args.base_dir,i['filepath'],i['filename']) for i in data}
with open(args.output_pickle, 'wb') as fd:
    pickle.dump(output, fd)

