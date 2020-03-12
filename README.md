# cromwell_for_ML
This is a wrapper around cromwell and WDL to enable you to run ML model at scale. \
In particular it runs jupyter notebook on google VM automatically 

# STILL TO DO:
1. visualization using jupyter notebook \
(Try to do during the code challenge week)

2. Maybe improve delocalization saving on a desired bucket using gsutil \
(Not urgent but possible, just make docker with gsutil in it)

3. Attach disk with data instead of localizing data from google bucket. \
(This is currently impossible. Even if you are root you can NOT run root command such as "sudo mount" unless the docker was launched with the special flag: "--cap-add=SYS_ADMIN". See https://stackoverflow.com/questions/36553617/how-do-i-mount-bind-inside-a-docker-container. My unsuccessful attempts are in the directory: MOUNT_DISK_NO_LOCALIZATION).  

### Installation
Type the following commands:
1. conda create --name trial python=3.6
2. conda activate trial
3. conda install -c conda-forge cromwell
4. git clone https://github.com/dalessioluca/cromwell_for_ML.git
5. cd cromwell_for_ML
6. edit the paramters.json file as necessary
7. ./submit_wdl_workflow.sh \
   (the first time you will be promped to specify the Cromwell server. \
    Type: https://cromwell-v47.dsde-methods.broadinstitute.org \
    Then run the command again)

### File descirption
1. submit_wdl_workflow.sh -> bash script which launches cromshell in a smart way
2. jupyter.wdl -> workflow in wdl 
3. main.ipynb -> example jupyter notebook you want to run
3. parameters.json -> all the parameters that the notebook needs to run

The end-user is responsible only of changing the jupyter notebook and the parameter file 


### Running Jupyter:

Let's compare the manual workflow and the WDL empowered workflow:

#### Manual Usage: 
You:
1. start a VM machine
2. git pull the code from your repository
3. copy the train_data and test_data inside the execution directory 
4. copy the parameters.json inside the execution directory
5. run the jupyter notebook from top-to-bottom
6. save the results into a google bucket
7. turn off the VM machine

#### WDL Usage
You:
1. on local machine edit the parameters.json as desired
2. run the command:\
   submit_wdl_workflow.sh jupyter.wdl parameters.json gs://ld-results-bucket/input_jsons 

   Here: 
   - jupyter.wdl is the file specifying the workflow and does not need to be changed
   - gs://ld-results-bucket/input_jsons is a bucket where the parameters file will be copied and the path_to_json will be passed to the workflow	
   
3. enjoy! The progress and results can be retieved with the commands: 
   - cromshell list -c -u
   - cromshell metadata
   - cromshell status
   
4. After sucessfull completion the results can be found:
   - in the cromwell execution bucket: broad-methods-cromwell-exec-bucket-v47/jupyter_localize
   - in the output bucket if specified in the parameters.json: for example gs://ld-results-bucket

### Assumptions:
To make this thing work you must ensure that:

1. the jupyter notebook:
   a. expect a file called parameters.json in the execution_dir
   b. produce outputs in execution_dir/dir_output
2. the file parameters.json:
   a. has few entries named "wdl.xxx" with the parameters for the wdl workflow
	- if "wdl.alias" is present the cromshell run will get that alias
	- if "wdl.bucket_output" is present the results will be copied from the default execution bucket to that bucket_output 
   b. has many other parameters with arbitrary nested structure to be read by the jupyter notebook
