source /home/users/fabricesch/remote_running/AutomatedPERTools/venv/bin/activate
rm false.out
echo start false
nohup python TrotterExample.py --variable False > false.out &
echo done