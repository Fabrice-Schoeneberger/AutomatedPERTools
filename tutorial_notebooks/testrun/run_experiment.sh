source /home/users/fabricesch/remote_running/AutomatedPERTools/venv/bin/activate
#rm true.out
rm false.out
#echo start true
#nohup python TrotterExample.py --variable True > true.out &
echo start false
nohup python TrotterExample.py --variable False > false.out &
echo done