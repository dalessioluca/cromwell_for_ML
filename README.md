# cromwell_for_ML
Cromwell_for_ML leverages *cromwell*, *WDL* and *Neptune* to train Machine Learning model at scale and nicely log the results for nice visualization and comparison. 
Possible use cases include:
1. hyperparameter optimization 
2. code development when multiple experiments are required.

In practice the solution boils down to running the command: 
> *./submit_neptune_ml.sh neptune_ml.wdl WDL_parameters.json --ml ML_parameters.json*

where:
1. _*submit_neptune_ml.sh*_ is a wrapper around cromshell
2. _*neptune_ml.wdl*_ is a WDL which specifies the following operations: \
	a. turn on a VM machine \
	b. checkout the correct version of the code from the github repository \
	c. launch the training of ML model \
	d. turn off the VM machine
3. _*WDL_parameters.json*_ contains few parameters such as the name of the git repository, and commit to use
4. _*ML_parameters.json*_ is a file with all the parameters necessary to specify the ML_model (learning_rate, etc)

In many situations the users should be able to only change the values in the _*WDL_parameters.json*_ and _*ML_parameters.json*_ to make the code run.

## Setup
To work you need to install both cromshell and Neptune.

### Neptune
1. Visit the website *https://neptune.ai/* and sign-up for a free account (sign-up bottom is in the top-right)
2. Run the jupyter notebook *AUXILIARY_FILES/TEST/test.ipynb*
3. If notebook executes sucesfully, then Neptune is installed and workingly properly

### Cromshell/Cromwell
1. Install *cromshell* (follow the instruction here: *https://github.com/broadinstitute/cromshell*)
2. If working remotely, connect to the Cisco Split-Tunnel VPN 
![split_VPN.png](https://github.com/dalessioluca/cromwell_for_ML/blob/master/AUXILIARY_FILES/PNG/split_VPN.png?raw=true)


3. Modify the file *AUXILIARY_FILES/TEST/test.json* to reflect your *NEPTUNE_API_TOKEN* and your *NEPTUNE_PROJECT* 
(use the same values you used in *AUXILIARY_FILES/test_neptune.ipynb*)
4. run the commands:
> *cd cromwell_for_ML/TEST* \
> *cromshell submit test.wdl test.json* \
> *cromshell list -u -c* \

You should see a list of all the runs submitted by cromshell. The last line should look like this:
![cromshell_list_test.png](https://github.com/dalessioluca/cromwell_for_ML/blob/master/AUXILIARY_FILES/PNG/cromshell_list_test.png?raw=true)
6. repeat the command *cromshell list -u -c* till you see the job has completed. 
At that point log into the neptune website *https://neptune.ai/* to see the results. 

### Cromshell and Neptune together
We are now going to use *cromshell* and *Neptune* to train a non-trivial ML model and log the results.
The conceptual overview is:
1. Cromshell will start a google Virtual Machine (VM) and localize all relevant files from google buckets to the VM 
2. on the VM we will checkout a github repo, and run the code *python main.py* which uses all the files we have localized to train a ML model
3. Neptune will log the metric
4. Cromshell turns of the VM

#### Preparation (one-time):
1. modify the first line of the file *SUBMIT/ML_parameters.json"* to reflect *your_neptune_username*,
1. modify the file */SUBMIT/LOCALIZED_FILES/credentials.json* by writing your own *NEPTUNE_API_TOKEN*
2. copy the files */SUBMIT/LOCALIZED_FILES/data_train.pt*, */SUBMIT/LOCALIZED_FILES/data_test.pt* and */SUBMIT/LOCALIZED_FILES/credentials.json* to your own google bucket, i.e.: 

> *gsutil -m cp SUBMIT/LOCALIZED_FILES/data_train.pt gs://my_bucket/data_train.pt* \
> *gsutil -m cp SUBMIT/LOCALIZED_FILES/data_test.pt gs://my_bucket/data_test.pt* \
> *gsutil -m cp SUBMIT/LOCALIZED_FILES/credentials.json gs://my_bucket/credentials.json*

3. modify the file */SUBMIT/WDL_parameters.json* to reflect the location where you copied the files *data_train.pt*, *data_train.pt* and *credentials.json* 
4. modify the first line on the file */SUBMIT/submit_neptune_ml.sh* to set your own google_bucket as the *DEFAULT_BUCKET*

Now we can finally train a ML model on the cloud and track all metrics using Neptune.

> *cd cromwell_for_ML/SUBMIT* \
> *./submit_neptune_ml.sh neptune_ml.wdl WDL_parameters.json --ml ML_parameters.json* \
> *cromshell list -u -c* 

The last row should list the run you just submitted and look like this (but listed as "Running" not "Succeded"):
![cromshell_list](https://github.com/dalessioluca/cromwell_for_ML/blob/master/AUXILIARY_FILES/PNG/cromshell_list_big_run.png?raw=true)

5. Log into the Neptune website and see your results streaming in. After a while your results should look like this:
![logged_metric](https://github.com/dalessioluca/cromwell_for_ML/blob/master/AUXILIARY_FILES/PNG/logged_metric.png?raw=true)

_Congrats you have trained your first ML model using *cromshell* and *Neptune*_ 


## How to use cromwell_for_ML to train YOUR model 
At the end of the day, you are going to run the command: 
*./submit_neptune_ml.sh neptune_ml.wdl WDL_parameters.json --ml ML_parameters.json* \

The file *neptune_ml.wdl* describes all operations which will happen on the VM. Namely:

1. localization of files
2. checking out the correct version of the code
3. running the python code

You can freely modify this code. For example you might want to localize more or less files or run a different python command. 
Changes to *neptune_ml.wdl* might require changes to *WDL_parameters.json*. 
Run the command:
> submit_neptune_ml.sh -t \
to see the template *WDL_parameters.json* for your modified 

The *WDL_parameters.json* contains:
1. the name of the git repository and commit you want to checkout 
2. the _locations_ of all files you want to localize from google buckets to VM machine. Among these file you always need the *credentials.json* (containing the NEPTUNE_API_TOKEN). You might or might not need the *data_train.pt* and *data_test.pt* files.  

The *ML_parameters.json* contains all the parameters for training your ML model. It will be automagically appear on the VM machine. It is up to you to make sure that your code reads and makes good use of the file *ML_parameters.json*

### Usefull commands:
cromshell list -c -u -> to check the status of the submitted runs
cromshell metadata ---> to retrive the the metadata of the last run. In particular the location of all log files 
cromshell status -----> retrive the status of the last run


