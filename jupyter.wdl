version 1.0

# Note that cromshell will automatically localize file in the following way:
# gs://ld-data-bucket/data/fashionmnist_test.pkl -> /cromwell_roo/data/fashionmnist_test.pkl

task parse_json {
    input {
            File input_json
            String pattern
           }

    command <<<

        python <<CODE

        import json

        # read the input json and write it back to have a copy
        with open("~{input_json}") as fp:
            my_dict = json.load(fp)

        # parse the dictionary for elements starting with the pattern and save it to a MAP output
        new_dict = {}
        for k,v in my_dict.items():
            if k.startswith("~{pattern}"):
                new_dict[k]=v
        print(json.dumps(new_dict))
        CODE
        >>>

    runtime {
        docker: "python"
        preemptible: 3
        }

    output {
        Map[String, String] output_map = read_json(stdout())
        }
}

task run_jupyter {
    input {
        File input_json

        File file_train
        File file_test
        File file_ckpt

        String dir_output
        String bucket_output

        String notebook_name
        String git_repo
        String commit_or_branch
    }

    command {
        # checkout the repo and the commit you want
        set -e
        git clone ~{git_repo} ./checkout_dir
        cd checkout_dir
    }
        #if ~{commit_or_branch} !- master:
        #    git checkout ~{commit_or_branch}
        #cp -r ./checkout_dir/* ../
        #cd ..
        #echo $(ls)
        # you are in the execution directory

        # (if necessary) rename input_json to parameters.json
        # This is what the notebook is expecting
        #if ~{input_json} != parameters.json:
        #    mv ~{input_json} parameters.json
        #echo $(ls)

        #run the notebook
        #pip install moviepy
        #pip install matplotlib
        #pip install jupyter_contrib_nbextensions
        #echo $(ls)
        #jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to=html --execute ~{notebook_name} --output main.html

    runtime {
        docker: "python"
        #docker: "us.gcr.io/broad-dsde-methods/luca_pyro"
        #bootDiskSizeGb: 50
        #memory: "15G"
        #cpu: 4
        #zones: "us-east1-d us-east1-c us-central1-a us-central1-c us-west1-b"
        #gpuCount: 1
        #gpuType: "nvidia-tesla-k80"
        #maxRetries: 0
    }

    output {
        File std_out = stdout()
        #File html_out = "main.html"
        #Array[File] results = glob("~{dir_output}/*")
    }
}


workflow jupyter_workflow {

 input {
   File parameters_json
 }

 call parse_json {
    input :
        input_json = parameters_json,
        pattern = "wdl."
 }

 call run_jupyter {
    input :
        input_json = parameters_json,

        file_train = parse_json.output_map["wdl.file_train"],
        file_test = parse_json.output_map["wdl.file_test"],
        file_ckpt = parse_json.output_map["wdl.file_ckpt"],

        dir_output = parse_json.output_map["wdl.dir_output"],
        bucket_output = parse_json.output_map["wdl.bucket_output"],

        notebook_name = parse_json.output_map["wdl.notebook_name"],
        git_repo = parse_json.output_map["wdl.git_repo"],
        commit_or_branch = parse_json.output_map["wdl.commit_or_branch"]
 }

 output {
    File out = run_jupyter.std_out
    #File html_out = run_jupyter.html_out
    #Array[File] results = run_jupyter.results
 }
}
