@echo off
scp -r -J fabricesch@alpha.lusi.uni-sb.de run_on_server.sh fabricesch@godot1:/localdisk/fabrice_remote_experiments/AutomatedPERTools/tutorial_notebooks/testrun/
::ssh fabricesch@alpha.lusi.uni-sb.de "ssh fabricesch@godot1 ""sh /localdisk/fabrice_remote_experiments/AutomatedPERTools/tutorial_notebooks/testrun/run_on_server.sh"""
::set "python=/localdisk/fabrice_remote_experiments/AutomatedPERTools/.venv/bin/python"
::set "TrotterExample=/localdisk/fabrice_remote_experiments/AutomatedPERTools/tutorial_notebooks/testrun/TrotterExample.py"
::set "baseline=/localdisk/fabrice_remote_experiments/AutomatedPERTools/tutorial_notebooks/testrun/"
::
::ssh fabricesch@alpha.lusi.uni-sb.de "ssh fabricesch@godot1 ""nohup %python% %TrotterExample% -a --pntsamples 256 --pntsinglesamples 4000 --onlyTomography --depths 2 4 8 16 32 64 --backend FakeVigoV2 > %baseline%Vigo-a_custom4k.out 2>&1"""
::ssh fabricesch@alpha.lusi.uni-sb.de "ssh fabricesch@godot1 ""nohup %python% %TrotterExample% -p --pntsamples 256 --pntsinglesamples 4000 --onlyTomography --depths 2 4 8 16 32 64 --backend FakeVigoV2 > %baseline%Vigo-a_custom4k.out 2>&1"""
::ssh fabricesch@alpha.lusi.uni-sb.de "ssh fabricesch@godot1 ""nohup %python% %TrotterExample% --pntsamples 256 --pntsinglesamples 4000 --onlyTomography --depths 2 4 8 16 32 64 --backend FakeVigoV2 > %baseline%Vigo-a_custom4k.out 2>&1"""


