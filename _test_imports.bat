@echo off
SET CONDA_PREFIX=C:\RAHUL\autom\predictpro_env
SET PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Library\mingw-w64\bin;%CONDA_PREFIX%\Library\usr\bin;%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\bin;%PATH%
SET PYTHONPATH=C:\RAHUL\autom\automated
echo Testing python version:
"%CONDA_PREFIX%\python.exe" --version
echo.
echo Running import tests:
"%CONDA_PREFIX%\python.exe" C:\RAHUL\autom\automated\_test_imports.py
