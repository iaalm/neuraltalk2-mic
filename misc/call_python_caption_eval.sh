#!/bin/bash

cd coco-caption
flock lock -c python myeval.py $1
cd ../
