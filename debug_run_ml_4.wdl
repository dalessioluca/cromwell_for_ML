version 1.0

# Note that cromshell will automatically localize file in the following way:
# gs://ld-data-bucket/data/fashionmnist_test.pkl -> /cromwell_roo/data/fashionmnist_test.pkl

task run_jupyter {
    input {

        File file_train
        File file_test
        File file_ckpt

        String notebook_name
        String git_repo
        String commit_or_branch

        String dir_output
        String bucket_output
        String input_json
    }

    command {
        # checkout the repo and the commit you want
        set -e
        git clone ~{git_repo} .  # it is important that you clone in the execution directory
        git checkout ~{commit_or_branch}

        # in the github there should be the file parameters_json
        mv ~{input_json} input.json

        #run the notebook
        pip install moviepy
        pip install matplotlib
        pip install jupyter_contrib_nbextensions
        jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to=html --execute ~{notebook_name} --output main.html
    }

    runtime {
        docker: "us.gcr.io/broad-dsde-methods/pyro:1.2.1"
        bootDiskSizeGb: 50
        memory: "15G"
        cpu: 4
        zones: "us-east1-d us-east1-c us-central1-a us-central1-c us-west1-b"
        gpuCount: 1
        gpuType: "nvidia-tesla-k80"
        maxRetries: 0
    }

    output {
        File html_out = "main.html"
        Array[File] results = glob("~{dir_output}/*")
    }
}


workflow wdl {

 input {
    File file_train
    File file_test
    File file_ckpt

    String notebook_name
    String git_repo
    String commit_or_branch

    String dir_output
    String bucket_output
    String input_json
 }

 call run_jupyter {
    input :

        file_train = file_train,
        file_test = file_test,
        file_ckpt = file_ckpt,

        notebook_name = notebook_name,
        git_repo = git_repo,
        commit_or_branch = commit_or_branch,

        dir_output = dir_output,
        bucket_output = bucket_output,
        input_json = input_json
 }

 output {
    File html_out = run_jupyter.html_out
    Array[File] results = run_jupyter.results
 }
}
