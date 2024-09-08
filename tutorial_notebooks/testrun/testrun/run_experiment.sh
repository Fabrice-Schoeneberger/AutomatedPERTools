#!/bin/bash

echo start
#source /localdisk/fabrice_remote_experiments/AutomatedPERTools/.venv/bin/activate

ulimit -n 10000

backends=("FakeVigoV2" "FakeMelbourneV2" "FakeCasablancaV2")
#extras=("-p" "-a" "" "-p -c" "-a -c" "-c")
extras=("-p -c" "-a -c" "-c")
#persamples=(10 100 500 1000 2000 5000 10000)
persamples=(1000)

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

for backend in "${backends[@]}"; do
    for persample in "${persamples[@]}"; do
        for extra in "${extras[@]}"; do
            if [ "$extra" != "" ]; then
                if [ "$persample" -ne 1000 ]; then
                    continue
                fi
            fi
            echo start ${name}_${value}_${extra}
            #rm ${name}_${value}_${extra}.out
            extra_without_spaces=$(echo "$extra" | sed 's/ /_/g')
            #python TrotterExample.py ${extra} --pntsamples 64 --pntsinglesamples 1000 --persamples ${persample} --backend ${backend} > ${backend}_${value}_${extra_without_spaces}.out &
            python TrotterExample.py ${extra} --pntsamples 64 --pntsinglesamples 1000 --onlyTomography --backend ${backend} > ${backend}_${value}_${extra_without_spaces}.out &
            #sleep 2 &
            #min=7
            #max=20
            #random_number=$((RANDOM % (max - min + 1) + min))
            #sleep $random_number &
            processes+=($!)
            n=${#processes[@]}
            while [ "$n" -gt 0 ]; do
                #date >> system_status_${backend}_${value}_${extra_without_spaces}.out
                #top -u fabricesch -n 1 -b -i -e g -E g >> system_status_${backend}_${value}_${extra_without_spaces}.out
                #echo "" >> system_status_${backend}_${value}_${extra_without_spaces}.out
                #date >> cpu_temp_${backend}_${value}_${extra_without_spaces}.out
                #sensors >> cpu_temp_${backend}_${value}_${extra_without_spaces}.out
                #echo "" >> cpu_temp_${backend}_${value}_${extra_without_spaces}.out
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
#nohup python TrotterExample.py -p --backend FakeCasablancaV2 > FakeCasablancaV2_0_-p.out &
#nohup python TrotterExample.py -s --backend FakeCasablancaV2 > FakeCasablancaV2_0_-s.out &
#nohup python TrotterExample.py --backend FakeCasablancaV2 > FakeCasablancaV2_0_.out &
#nohup python TrotterExample.py -p --backend FakeMelbourneV2 > FakeMelbourneV2_0_-p.out &
#nohup python TrotterExample.py -s --backend FakeMelbourneV2 > FakeMelbourneV2_0_-s.out &
#nohup python TrotterExample.py --backend FakeMelbourneV2 > FakeMelbourneV2_0_.out &
