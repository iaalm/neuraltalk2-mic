#!/bin/bash

cd coco-caption
flock lock python myeval.py $1
cd ../
