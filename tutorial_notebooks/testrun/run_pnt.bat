@echo off
setlocal enabledelayedexpansion

echo start
::call "C:\localdisk\fabrice_remote_experiments\AutomatedPERTools\.venv2\Scripts\activate"

set pntsamples=4 8 16 32 64 128
set pntsinglesamples=10 50 100 200 500 1000 2000 5000


SET /A i=0
SET /A ip=1
:forloop
SET /A j=0
echo %i%
:loop
SET /A j = %j%+%ip%
set count=0

:: Count how many instances of TrotterExample.py are running
for /f %%a in ('wmic process where "name='python.exe'" get CommandLine ^| find /c "PER.py"') do set count=%%a

:: Display the count
echo Currently running instances: %count%, times is at %j%

:: If fewer than 5 instances remain, exit loop and continue
if %count% LSS 1 (
    goto :continueScript
)

:: Wait 10 seconds before checking again
timeout /t 20 >nul
goto loop

:continueScript

if %i% LSS 2 (
    start /B python PER.py --noise_strengths 0.5 1 2 --persamples 1000 > output/output6_%i%.log 2>&1
    start /B python PER.py --noise_strengths 0.5 1 2 --persamples 1000 > output/output61_%i%.log 2>&1
    start /B python PER.py --noise_strengths 0.5 1 2 --persamples 1000 > output/output62_%i%.log 2>&1
    start /B python PER.py --noise_strengths 0.5 1 2 --persamples 1000 > output/output63_%i%.log 2>&1
)

SET /A i = %i%+%ip%
if %i% LSS 100 (
    goto forloop
)

start /B python PER.py --noise_strengths 0.5 1 2 --persamples 1000 > output/output6_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 --pntsamples 16 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output0_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 16 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output2_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 64 --pntsamples 16 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output3_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 64 128 --pntsamples 16 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output3_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 64 128 256 --pntsamples 16 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output3_%i%.log 2>&1

::start /B python TrotterExample.py --depths 2 4 8 --pntsamples 64 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output0_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 64 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output2_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 64 --pntsamples 64 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output3_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 64 128 --pntsamples 64 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output3_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 64 128 256 --pntsamples 64 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output3_%i%.log 2>&1


::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 4 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output0_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 8 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 32 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output3_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 64 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output4_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 128 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output5_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 256 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output6_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 512 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output6_%i%.log 2>&1


::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 4 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output0_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 8 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 16 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output2_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 32 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output3_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 128 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output5_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 256 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output6_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 512 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output6_%i%.log 2>&1


::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 10 --onlyTomography --num_qubits 4 > output/output0_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 20 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 30 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 40 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 50 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 60 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 70 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 80 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 90 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 200 --onlyTomography --num_qubits 4 > output/output3_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 500 --onlyTomography --num_qubits 4 > output/output4_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 1000 --onlyTomography --num_qubits 4 > output/output5_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 2000 --onlyTomography --num_qubits 4 > output/output6_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 --pntsamples 16 --pntsinglesamples 5000 --onlyTomography --num_qubits 4 > output/output7_%i%.log 2>&1


::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 10 --onlyTomography --num_qubits 4 > output/output0_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 20 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 30 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 40 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 50 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 60 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 70 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 80 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 90 --onlyTomography --num_qubits 4 > output/output1_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 100 --onlyTomography --num_qubits 4 > output/output2_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 200 --onlyTomography --num_qubits 4 > output/output3_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 500 --onlyTomography --num_qubits 4 > output/output4_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 2000 --onlyTomography --num_qubits 4 > output/output6_%i%.log 2>&1
::start /B python TrotterExample.py --depths 2 4 8 16 32 --pntsamples 64 --pntsinglesamples 5000 --onlyTomography --num_qubits 4 > output/output7_%i%.log 2>&1

