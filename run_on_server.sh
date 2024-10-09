cd /localdisk/fabrice_remote_experiments/AutomatedPERTools/tutorial_notebooks/testrun
source /localdisk/fabrice_remote_experiments/AutomatedPERTools/.venv/bin/activate
nohup python process_experiment_pickle.py --pntsamples 256 --pntsinglesamples 4000 --onlyTomography --depths 2 4 8 16 32 64 --backend FakeCasablancaV2 > Casablanca_custom4k_Saved.out 2>&1 &
echo started 1
nohup python process_experiment_pickle.py --pntsamples 256 --pntsinglesamples 4000 --onlyTomography --depths 2 4 8 16 32 64 --backend FakeMelbourneV2 > Melbourne_custom4k_Saved.out 2>&1 &
echo started 2
#nohup python process_experiment_pickle.py --pntsamples 256 --pntsinglesamples 4000 --onlyTomography --depths 2 4 8 16 32 64 --backend FakeVigoV2 > Vigo_custom4k_Saved.out 2>&1 &
#echo started 3
#nohup python process_experiment_pickle.py --backend FakeCasablancaV2 > Casablanca_custom_Saved.out 2>&1 &
#echo started 4
#nohup python process_experiment_pickle.py --backend FakeMelbourneV2 > Melbourne_custom_Saved.out 2>&1 &
#echo started 5
#nohup python process_experiment_pickle.py --backend FakeVigoV2 > Vigo_custom_Saved.out 2>&1 &
#echo started 6