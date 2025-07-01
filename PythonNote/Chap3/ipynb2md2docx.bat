@echo off
rem This script converts all .ipynb files in the current directory to .docx files

rem Check if jupyter and pandoc are installed
where jupyter >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Jupyter is not installed. Please install Jupyter and try again.
    pause
    exit /b
)

where pandoc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Pandoc is not installed. Please install Pandoc and try again.
    pause
    exit /b
)

rem Loop through all .ipynb files in the current directory
for %%f in (*.ipynb) do (
    echo Converting %%f to markdown...
    jupyter nbconvert --to markdown "%%f"
    
    set "filename=%%~nf"
    echo Converting %%~nf.md to %%~nf.docx...
    pandoc "%%~nf.md" -o "%%~nf.docx"
    
    rem Optional: delete the intermediate markdown file
    del "%%~nf.md"
)

echo Conversion complete.
pause
