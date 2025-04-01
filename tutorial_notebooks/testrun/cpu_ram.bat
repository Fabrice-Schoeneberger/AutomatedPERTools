@echo off
setlocal enabledelayedexpansion
set /A i = 6

start /B python PER.py --noise_strengths 0.5 1 2 --persamples 4000 > output/output666_%i%.log 2>&1