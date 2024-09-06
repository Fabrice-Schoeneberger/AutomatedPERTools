@echo off
setlocal

REM Führt das PowerShell-Skript aus, um den Dateiauswahldialog anzuzeigen und speichert den Dateipfad in der Variablen filepath
for /f "delims=" %%I in ('powershell -command "Add-Type -AssemblyName System.windows.forms; $ofd = New-Object System.Windows.Forms.OpenFileDialog; $ofd.Filter = 'Alle Dateien (*.*)|*.*'; $ofd.ShowDialog() | Out-Null; $ofd.FileName"') do set "filepath=%%I"

REM Gibt den ausgewählten Dateipfad aus
echo Ausgewählter Dateipfad: %filepath%

REM Hier kann der Dateipfad weiterverarbeitet werden
scp %filepath% fabricesch@alpha.lusi.uni-sb.de:/home/users/fabricesch/temp/
