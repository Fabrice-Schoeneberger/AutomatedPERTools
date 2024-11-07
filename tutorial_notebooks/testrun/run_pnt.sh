#!/bin/bash

echo start
source /localdisk/fabrice_remote_experiments/AutomatedPERTools/.venv2/bin/activate

ulimit -n 10000

pntsamples=(4 8 16 32 64 128)
pntsinglesamples=(10 50 100 200 500 1000 2000 5000)
depths=("2 4 8" "2 4 8 16" "2 4 8 16 32" "2 4 8 16 32 64")

for depth in "${depths[@]}"; do
    nohup python TrotterExample.py --depths ${depth} --pntsamples 16 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > depths_16_100.out &
done

sleep 3600

for depth in "${depths[@]}"; do
    python TrotterExample.py --depths ${depth} --pntsamples 64 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > depths_64_1000.out &
done

sleep 3600

for sample in "${pntsamples[@]}"; do
    python TrotterExample.py --depths 2 4 8 16 --pntsamples ${sample} --pntsinglesamples 100 --onlyTomography --num_qubits 4 > 2_4_8_16_${sample}_100.out &
done

sleep 3600

for sample in "${pntsamples[@]}"; do
    python TrotterExample.py --depths 2 4 8 16 32 --pntsamples ${sample} --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > 2_4_8_16_32_${sample}_1000.out &
done

sleep 3600

for sample in "${pntsinglesamples[@]}"; do
    python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples ${sample} --onlyTomography --num_qubits 4 > 2_4_8_16_16_${sample}.out &
done

sleep 3600

for sample in "${pntsinglesamples[@]}"; do
    python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples ${sample} --onlyTomography --num_qubits 4 > 2_4_8_16_32_64_${sample}.out &
done