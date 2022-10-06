#!/usr/bin/env bash
for i in {0..5}
do 
    python my_latency.py ourvisiontransformer_b${i} > ourvisiontransformer_b${i}.txt
    python my_latency.py ourchannel_b${i} > ourchannel_b${i}.txt
done