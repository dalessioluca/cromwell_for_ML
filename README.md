# cromwell_for_ML
Examples of how to use cromwell to run ML model at scale 


### STILL TO DO:
delocalization saving on a desired bucket
visualization using jupyter notebook

### File descirption
1. main.ipynb -> jupyter notebook you want to run
2. parameters.json -> all the parameters that the notebook needs to run
3. wdl_inputs.json -> json with a single entry specifying where, in the cloud, the input_parameters file is 
4. jupyter.wdl -> workflow in wdl 


### Manual Usage
You:
1. start a VM machine
2. git pull the code from your repository
3. copy the train_data and test_data inside the execution directory 
4. copy the parameters.json inside the execution directory
5. run the jupyter notebook from top-to-bottom
6. save the results into a google bucket
7. turn off the VM machine

### WDL Usage
You:
1. on local machine edit the parameters.json as desired
2. run the command:

   submit_wdl_workflow.sh jupyter.wdl parameters.json gs://ld-results-bucket/input_jsons
   
3. enjoy! The progress and results can be retieved with the commands:

   a cromshell list -c -u
   
   b cromshell metadata

### Assumptions:
You have installed: cromshell, BLABLA

1. the jupyter notebook:
   a. expect a file called parameters.json in the execution_dir
   b. produce outputs in execution_dir/dir_output
2. the file parameters.json:
   a. has few entries named "wdl.xxx" with the parameters for the wdl workflow
   b. has many opther parameters with arbitrary nested structure to be read by the jupyter notebook
