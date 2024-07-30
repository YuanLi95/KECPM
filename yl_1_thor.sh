#! /bin/bash

for ddir in  "no_none_unified_tags_txt/seed_97"
do
CUDA_VISIBLE_DEVICES=1 python main.py --batch_size=2 -pt=imagecap_know -kn=1 -ddir=$ddir --model_path=t5-base --model_size=base

done
