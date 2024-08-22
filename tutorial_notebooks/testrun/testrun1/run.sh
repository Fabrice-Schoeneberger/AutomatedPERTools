#!/bin/bash

echo start
source /localdisk/fabrice_remote_experiments/AutomatedPERTools/.venv/bin/activate

names=("FakeMelbourneV2" "FakeCasablancaV2")
do1000=(1)
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
            #rm ${name}_${value}_${extra}.out
            nohup python TrotterExample.py ${extra} ${extras2} --backend ${name} > ${name}_${value}_${extra}.out &
            #sleep 2 &
            #min=7
            #max=20
            #random_number=$((RANDOM % (max - min + 1) + min))
            #sleep $random_number &
            processes+=($!)
            n=${#processes[@]}
            while [ "$n" -gt 0 ]; do
                date >> system_status_${name}_${value}_${extra}.out
                top -u fabricesch -n 1 -b -i -e g -E g >> system_status_${name}_${value}_${extra}.out
                echo "" >> system_status_${name}_${value}_${extra}.out
                date >> cpu_temp_${name}_${value}_${extra}.out
                sensors >> cpu_temp_${name}_${value}_${extra}.out
                echo "" >> cpu_temp_${name}_${value}_${extra}.out
                sleep 300
                remove_list=()
                is_done=0
                for process in "${processes[@]}"; do
                    if ps -p "$process" > /dev/null; then
                        sleep 0
                    else
                        is_done=1
                        remove_list+=($process)
                    fi
                done

                if [ "$is_done" -eq 1 ]; then
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
                fi
            done


            #if [ "$n" -gt 0 ]; then
            #    wait -n ${processes[@]}
            #fi
        done
    done
done

#For manuell entry:
# FakeCasablancaV2
# python TrotterExample.py --pntsamples 1 --pntsinglesamples 1 --persamples 1 --backend FakeMelbourneV2
# nohup python TrotterExample.py --backend FakeMelbourneV2 > error.out &
