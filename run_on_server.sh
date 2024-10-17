cd /localdisk/fabrice_remote_experiments/AutomatedPERTools/tutorial_notebooks/testrun
source /localdisk/fabrice_remote_experiments/AutomatedPERTools/.venv/bin/activate
nohup python TrotterExample.py --pntsamples 256 --pntsinglesamples 4000 --onlyTomography --depths 2 4 8 16 32 64 --backend FakeCasablancaV2 > CNOT_LAYER.out 2>&1 &
echo started 1
