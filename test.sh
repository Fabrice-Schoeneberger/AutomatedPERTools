echo parent process $$ starting
sleep 30 &
pid1=$!
echo $pid1
sleep 1
sleep 60 &
pid2=$!
echo $pid2
sleep 1
sleep 100 &
pid3=$!
echo $pid3
wait $pid1 $pid2
echo parent wait finished, exit status $?