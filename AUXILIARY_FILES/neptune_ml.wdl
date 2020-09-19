version 1.0

# Note that cromshell will automatically localize file in the following way:
# gs://ld-data-bucket/data/fashionmnist_test.pkl -> /cromwell_roo/data/fashionmnist_test.pkl

task train {
    input {
        File ML_parameters
        File data_train
        File data_test
        File credentials_json
        String git_repo
        String git_branch_or_commit
    }


    command <<<
        echo "START --> Content of exectution dir"
        echo $(ls)
        exec_dir=$(pwd)
        echo "--> $exec_dir"
        
        # 1. checkout the repo in the EXCUTION DIRECTORY
        # need to checkout in a different directory and then copy to execution directory
        set -e
        git clone ~{git_repo} ./checkout_dir
        cd ./checkout_dir
        git checkout ~{git_branch_or_commit}
        cp -r ./* $exec_dir/
        cd $exec_dir
        echo "AFTER GIT --> Content of exectution dir"
        echo $(ls)
        
        # 2. link the file which have been localized to the execution directory 
        # and give them the name the main.py expects
        ln -s ~{ML_parameters} ./ML_parameters.json
        ln -s ~{data_train} ./data_train.pt
        ln -s ~{data_test} ./data_test.pt
        ln -s ~{credentials_json} ./credentials.json
        echo "AFTER CHANGING NAMES --> Content of exectution dir"
        echo $(ls)

        # 3. run python code only if credentials are correct
        token=$(cat credentials.json | jq .NEPTUNE_API_TOKEN)
        echo $token
        if [ $token != 'null' ]; then 
            export NEPTUNE_API_TOKEN=$token
            pip install neptune-client 
            pip install psutil
            python main.py 
        fi

    >>>
    
    runtime {
          docker: "python"
          cpu: 1
          preemptible: 3
    }
    
#    runtime {
#         docker: "us.gcr.io/broad-dsde-methods/pyro_matplotlib:0.0.2"
#         bootDiskSizeGb: 50
#         memory: "26G"
#         cpu: 4
#         zones: "us-east1-d us-east1-c"
#         gpuCount: 1
#         gpuType: "nvidia-tesla-k80" #"nvidia-tesla-p100" 
#         maxRetries: 0
#         preemptible_tries: 0
#    }

}

workflow neptune_ml {

    input {
        File ML_parameters 
        File data_train 
        File data_test
        File credentials_json 
        String git_repo
        String git_branch_or_commit 
    }

    call train { 
        input :
            ML_parameters = ML_parameters,
            credentials_json = credentials_json,
            data_train = data_train,
            data_test = data_test,
            git_repo = git_repo,
            git_branch_or_commit = git_branch_or_commit
    }
}
