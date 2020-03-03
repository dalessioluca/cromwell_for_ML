# cromwell_for_ML
Examples of how to use cromwell to run ML model at scale 

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
2. copy parameters.json somewhere in a google bucket
3. edit wdl_inputs.json with the location of the file you just edited
4. from your local machine run the command:

   cromshell submit jupyter.wdl wdl_input.json 

5. enjoy! The progress and results can be retieved with the commands:

   a cromshell list -c -u
   
   b cromshell metadata

