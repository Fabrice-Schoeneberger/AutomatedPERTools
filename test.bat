@echo off
::ssh fabricesch@alpha.lusi.uni-sb.de "ssh fabricesch@godot1 ""python /localdisk/fabrice_remote_experiments/AutomatedPERTools/tutorial_notebooks/testrun/test.py > /localdisk/fabrice_remote_experiments/AutomatedPERTools/tutorial_notebooks/testrun/test.out &"""
ssh fabricesch@alpha.lusi.uni-sb.de "ssh fabricesch@godot1 ""nohup python /localdisk/fabrice_remote_experiments/AutomatedPERTools/tutorial_notebooks/testrun/test.py > /localdisk/fabrice_remote_experiments/AutomatedPERTools/tutorial_notebooks/testrun/test.out 2>&1 &"""
