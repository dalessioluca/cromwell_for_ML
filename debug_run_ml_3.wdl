version 1.0

# Note that cromshell will automatically localize file in the following way:
# gs://ld-data-bucket/data/fashionmnist_test.pkl -> /cromwell_root/ld-data-bucket/data/fashionmnist_test.pkl
# json_wdl_inputs.json -> /cromwell_root/json_wdl_inputs.json

task parse_input_json {
    input {
            File input_file
            String pattern
           }
   parameter_meta {
        input_file: {
            localization_optional: true
            }
        }

    command <<<
        python <<CODE

        import json

        # read the input json as a dictionary
        new_dict = {}
        with open("~{input_file}") as fp:
            my_dict = json.load(fp)
            for k,v in my_dict.items():
                if k.startswith("~{pattern}"):
                    new_dict[k]=v

        # set wdl.active to TRUE
        new_dict["wdl.active"] = False
        my_dict["wdl.active"] = False

        # print to stdout and on file
        print(json.dumps(new_dict))
        with open("all_params.json", 'w') as fp:
            json.dump(my_dict, fp)
        CODE
        >>>

    output {
        Map[String, String] output_map = read_json(stdout())
        File all_params_json = "parameters.json"
        }
}

task run_jupyter {
    input {
        File params
        File train_data
        File test_data
        File checkpoint
        String notebook_name
        String git_repo
        String commit_or_branch
        String bucket_output
    }

    command {
        # checkout the repo and the commit you want
        set -e
        git clone ~{git_repo} .
        git checkout ~{commit_or_branch}
        # at this point you are in the execution directory

        #run the notebook
        pip install moviepy
        pip install matplotlib
        pip install jupyter_contrib_nbextensions
        jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to=html --execute ~{notebook_name} --output main.html
    }

    output {
        File html_out = "main.html"
        Array[File] results = glob("results/*")
    }
}


workflow luca_workflow {

 input {
   File parameters_json
 }

 call parse_input_json {
    input :
        input_file = parameters_json,
        pattern = "wdl."
 }

 call run_jupyter {
    input :
        params = parse_input_json.all_params_json,
        train_data = parse_input_json.output_map["wdl.train_data"],
        test_data = parse_input_json.output_map["wdl.test_data"],
        checkpoint = parse_input_json.output_map["wdl.checkpoint"],
        notebook_name = parse_input_json.output_map["wdl.notebook_name"],
        git_repo = parse_input_json.output_map["wdl.git_repo"],
        commit_or_branch = parse_input_json.output_map["wdl.commit_or_branch"],
        bucket_output = parse_input_json.output_map["wdl.bucket_output"]
 }

 output {
    File out = parse_input_json.all_params_json
    File html = run_jupyter.html_out
    Array[File] results = run_jupyter.results
 }
}
