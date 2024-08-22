#
p=()
sleep 1 &
p+=($!)
echo ${p[@]}
sleep 10 &
p+=($!)
echo ${p[@]}
sleep 1
r=()
for process in "${p[@]}"; do
    if ps -p "$process" > /dev/null; then
	sleep 0
    else
	r+=($process)
    fi
done
echo ${r[@]}
