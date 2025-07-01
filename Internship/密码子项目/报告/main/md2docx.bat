@echo off
rem This script converts all .md files in the current directory to .docx files

rem Check if pandoc is installed
where pandoc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Pandoc is not installed. Please install Pandoc and try again.
    pause
    exit /b
)

rem Loop through all .md files in the current directory
for %%f in (*.md) do (
    echo Converting %%f to %%~nf.docx...
    pandoc "%%f" -o "%%~nf.docx"
    
    rem Optional: delete the intermediate markdown file
    rem del "%%f"
)

echo Conversion complete.
pause