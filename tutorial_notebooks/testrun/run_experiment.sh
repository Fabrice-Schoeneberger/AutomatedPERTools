echo start
source /localdisk/fabrice_remote_experiments/AutomatedPERTools/.venv/bin/activate
rm truetrue.out
rm truefalse.out
rm falsefalse.out
echo start truetrue
nohup python TrotterExample.py --plusone True --sum True --backend FakeCasablancaV2 > truetrue.out &
echo start truefalse
nohup python TrotterExample.py --plusone True --sum False --backend FakeCasablancaV2 > truefalse.out &
echo start falsefalse
nohup python TrotterExample.py --plusone False --sum False --backend FakeCasablancaV2 > falsefalse.out &

rm 1000truetrue.out
rm 1000truefalse.out
rm 1000falsefalse.out
echo start 1000 truetrue
nohup python TrotterExample.py --plusone True --sum True --pntsamples 64 --pntsinglesamples 1000 --persamples 1000 --backend FakeCasablancaV2 > 1000truetrue.out &
echo start 1000 truefalse
nohup python TrotterExample.py --plusone True --sum False --pntsamples 64 --pntsinglesamples 1000 --persamples 1000 --backend FakeCasablancaV2 > 1000truefalse.out &
echo start 1000 falsefalse
nohup python TrotterExample.py --plusone False --sum False --pntsamples 64 --pntsinglesamples 1000 --persamples 1000 --backend FakeCasablancaV2 > 1000falsefalse.out &
echo done