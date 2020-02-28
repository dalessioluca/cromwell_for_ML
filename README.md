# cromwell_for_ML
Examples of how to use cromwell to run ML model at scale 

The idea is to have a smooth transition from development on local machine to simulation at scale on the cloud.

The inputs are:
1. jupyter notebook to run
2. input_parameter.json

The oputputs are:
1. the html version of the jupyter notebook with the results
2. a lot of png and gif files
3. a copy of the input_parameter.json
4. a log file

On local machine this notebook can be run with the commands:
1. pip install jupyter_contrib_nbextensions
2. jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to=html --execute main.ipynb --output main.html

On the cloud this will be run using a WDL and cromshell
