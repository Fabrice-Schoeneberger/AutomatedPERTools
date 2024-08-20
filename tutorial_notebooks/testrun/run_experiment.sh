#!/bin/bash

echo start
source /localdisk/fabrice_remote_experiments/AutomatedPERTools/.venv/bin/activate

names=("FakeMelbourneV2" "FakeCasablancaV2")
do1000=(0 1)
extras1=("-p" "-s" "")

#Do not touch this:
processes=()

is_in_array() {
    local item="$1"
    local array=("${!2}")
    
    for elem in "${array[@]}"; do
        if [ "$elem" -eq "$item" ]; then
            return 0  # Found
        fi
    done
    return 1  # Not found
}

for value in "${do1000[@]}"; do
    for name in "${names[@]}"; do
        #This iterates over how many samples to use
        if [ "$value" -eq 0 ]; then
            extras2=""
        else
            extras2="--pntsamples 64 --pntsinglesamples 1000 --persamples 1000"
        fi
        #This checks which processes have already finished and removes them from the list
        for extra in "${extras1[@]}"; do
            echo start ${name}_${value}_${extra}
            rm ${name}_${value}_${extra}.out
            nohup python TrotterExample.py ${extra} ${extras2} --backend ${name} > ${name}_${value}_${extra}.out &
            #min=7
            #max=20
            #random_number=$((RANDOM % (max - min + 1) + min))
            #sleep $random_number &
            processes+=($!)
            remove_list=()
            for process in "${processes[@]}"; do
                if ps -p "$process" > /dev/null; then
                    sleep 0
                else
                    remove_list+=($process)
                fi
            done

            new_processes=()
            for item in "${processes[@]}"; do
                if ! is_in_array "$item" remove_list[@]; then
                    new_processes+=("$item")
                fi
            done
            processes=()
            for p in ${new_processes[@]}; do
                processes+=($p)
            done
            n=${#processes[@]}
            if [ "$n" -gt 2 ]; then
                wait -n ${processes[@]}
            fi
        done
    done
done