cd /localdisk/fabrice_remote_experiments/AutomatedPERTools/tutorial_notebooks/testrun
source /localdisk/fabrice_remote_experiments/AutomatedPERTools/.venv/bin/activate
nohup python TrotterExample.py --pntsamples 256 --pntsinglesamples 4000 --onlyTomography --depths 2 4 8 16 32 64 --backend FakeCasablancaV2 > Casablanca_custom4k.out 2>&1 &
echo started 1
nohup python TrotterExample.py --pntsamples 256 --pntsinglesamples 4000 --onlyTomography --depths 2 4 8 16 32 64 --backend FakeMelbourneV2 > Melbourne_custom4k.out 2>&1 &
echo started 2
nohup python TrotterExample.py --pntsamples 256 --pntsinglesamples 4000 --onlyTomography --depths 2 4 8 16 32 64 --backend FakeVigoV2 > Vigo_custom4k.out 2>&1 &
echo started 3