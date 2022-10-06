#!/bin/bash
for i in {0..5}
do
    echo "generating flops and params for ourvisiontransformer${i}\n"
    python new_get_flops.py ourvisiontransformer_b${i} > flops_and_param_${i}.txt
    echo "successfully!\n"
done