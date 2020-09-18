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
        apt-get install git
        echo "Content of exectution dir"
        echo $(ls)

        # 1. checkout the repo and the commit you want in the CHECKOUT_DIR
        set -e
        git clone ~{git_repo} ./checkout_dir
        cd checkout_dir
        git checkout ~{git_branch_or_commit}

        # 2. File are localized in the EXECUTION_DIR while code is in the CHECKOUT_DIR
        # Copy everything from checkout to execution dir
        cp -r ./* ../  # from CHECKOUT_DIR to EXECUTION_DIR
        cd ..          # move to EXECUTION_DIR
        echo "Content of exectution dir"
        echo $(ls)

        # 3. change name contained the data to data_train and data_test (this is what main.py is expecxting)
        name_credentials=$(basename ~{credentials_json})
        name_train=$(basename ~{data_train})
        name_test=$(basename ~{data_test})
        
        echo "----->"
        echo ~{credentials_json}
        echo $name_credentials
        echo "----->"

        mv $name_train data_train.pt
        mv $name_test data_test.pt
        mv $name_credentials credentials.json

        # 4. check content of directory
        echo "in the current directory there is"
        echo $(ls)

        # 5. run python code only if credentials are correct
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
