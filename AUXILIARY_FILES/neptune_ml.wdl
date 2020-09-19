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
        
        # 1. checkout the repo in the EXCUTION DIRECTORY
        set -e
        git clone ~{git_repo} ./checkout_dir
        cd ./checkout_dir
        git checkout ~{git_branch_or_commit}
        cp -r ./* ../
        cd ../
        echo "AFTER GIT --> Content of exectution dir"
        echo $(ls)
        
        # 2. change change the file names to what the code expect 
        name_ML_parameters=$(basename ~{ML_parameters})
        name_credentials=$(basename ~{credentials_json})
        name_train=$(basename ~{data_train})
        name_test=$(basename ~{data_test})
        echo "DEBUG" $name_test ~{data_test}

        ln -s $name_ML_parameters ML_parameters.json
        ln -s $name_train data_train.pt
        ln -s $name_test data_test.pt
        ln -s $name_credentials credentials.json
        echo "AFTER CHANGING NAMES --> Content of exectution dir"
        echo $(ls)

        # 3. run python code only if credentials are correct
        token=$(cat credentials.json | jq .NEPTUNE_API_TOKEN)
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
