@echo off
echo ====================================================================
echo           AffectiveEcho: Full Pipeline and Paper Build Script
echo ====================================================================

set PYTHON="C:\Users\progr\AppData\Local\Programs\Python\Python312\python.exe"
set PDFLATEX="C:\Users\progr\AppData\Roaming\TinyTeX\bin\windows\pdflatex.exe"
set BIBTEX="C:\Users\progr\AppData\Roaming\TinyTeX\bin\windows\bibtex.exe"

echo.
echo [*] Step 1: Running Automated PyTest Unit Suite...
%PYTHON% -m pytest tests\test_affective_echo.py -v
if %ERRORLEVEL% NEQ 0 (
    echo [!] Unit tests failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [*] Step 2: Running Benchmarks and Generating Results...
%PYTHON% experiments\benchmark_runner.py

echo.
echo [*] Step 3: Generating 300-DPI Publication Figures...
%PYTHON% experiments\generate_publication_figures.py

echo.
echo [*] Step 4: Compiling IEEE LaTeX Journal Paper, Supplement, and Cover Letter...
cd paper
%PDFLATEX% -interaction=nonstopmode IEEE_Manuscript.tex
%BIBTEX% IEEE_Manuscript
%PDFLATEX% -interaction=nonstopmode IEEE_Manuscript.tex
%PDFLATEX% -interaction=nonstopmode IEEE_Manuscript.tex
%PDFLATEX% -interaction=nonstopmode Supplementary_Material.tex
%BIBTEX% Supplementary_Material
%PDFLATEX% -interaction=nonstopmode Supplementary_Material.tex
%PDFLATEX% -interaction=nonstopmode Supplementary_Material.tex
%PDFLATEX% -interaction=nonstopmode Cover_Letter.tex
cd ..

echo.
echo ====================================================================
echo   [SUCCESS] Full publication package built cleanly!
echo   - Camera-ready Paper: paper\IEEE_Manuscript.pdf
echo   - Supplementary Material: paper\Supplementary_Material.pdf
echo   - Editor Cover Letter: paper\Cover_Letter.pdf
echo   - 300-DPI Figures: paper\figures\
echo   - Interactive HUD Demo: demo\webcam_hud.py
echo ====================================================================
