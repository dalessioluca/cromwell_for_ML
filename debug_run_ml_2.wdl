version 1.0

# Note that cromshell will automatically localize file in the following way:
# gs://ld-data-bucket/data/fashionmnist_test.pkl -> /cromwell_root/ld-data-bucket/data/fashionmnist_test.pkl
# json_wdl_inputs.json -> /cromwell_root/json_wdl_inputs.json

import parse_json.py

task read_file {
   input {
        File input_file
   }
   parameter_meta {
        input_file: {
            localization_optional: true
        }
    }
    command {
        set -e
        cat ~{input_file}
    }
    output {
        File output_file=stdout()
    }
}

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
    command {
        python parse_json.py ~{input_file} ~{pattern}
        }
    output {
        Map[String, String] my_map = read_json(stdout())
        File output_file = "./params.json"
        }
}


task parse_file_to_map {
    input {
            File input_file
            String pattern
           }
   parameter_meta {
        input_file: {
            localization_optional: true
            }
        }
    command {
        set -e

    echo '{"foo":"bar"}'
  >>>
  output {
    Map[String, String] my_map = read_json(stdout())



        egrep ~{pattern}  ~{input_file}
        read_int(stdout())
        filtered_map = { egrep ~{pattern}  ~{input_file} }
        }
    output {
        #Map[String, String] map=read_json(stdout())
        }
}

task run_jupyter {
    input {
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
        # at this point you are in the working directory: /cromwell_root

        #run the notebook
        pip install moviepy
        pip install matplotlib
        pip install jupyter_contrib_nbextensions
        jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to=html --execute ~{notebook_name} --output main.html
    }

    output {
        #File html_out = "checkout_dir/main.html"
        #Array[File] results = glob("checkout_dir/results/*")
        #File outputs = "inputs"
        File output_file=stdout()
    }
}




workflow luca_workflow {
 input {
   File parameters_json
 }

 call read_file {
    input :
        input_file = parameters_json
 }

 call parse_file_to_map {
    input :
        input_file = parameters_json,
        pattern = "wdl."
 }

### call run_jupyter {
###    input :
###        train_data = parse_file_to_map.map["wdl.train_data"],
###        test_data = parse_file_to_map.map["wdl.test_data"],
###        checkpoint = parse_file_to_map.map["wdl.checkpoint"],
###        notebook_name = parse_file_to_map.map["wdl.notebook_name"],
###        git_repo = parse_file_to_map.map["wdl.git_repo"],
###        commit_or_branch = parse_file_to_map.map["wdl.commit_or_branch"],
###        bucket_output = parse_file_to_map.map["wdl.bucket_output"]
### }

 output {
    File out = parse_file_to_map.map
 }
}


#####
#####task read_input {
#####  input {
#####    File json_inputs
#####  }
#####  output {
#####  }
#####
#####  command {
#####        echo "Hello world"
#####        inputs = read_json(json_inputs)
#####        echo inputs
#####
#####
######        String notebook_name
######        String git_repo
######        String commit_or_branch
######        String bucket_output
######
######     # checkout the repo and the commit you want
######     set -e
######     git clone ~{git_repo} .
######     git checkout ~{commit_or_branch}
######     # at this point you are in the working directory: /cromwell_root
######
######     #run the notebook
######     pip install moviepy
######     pip install matplotlib
######     pip install jupyter_contrib_nbextensions
######     jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to=html --execute ~{notebook_name} --output main.html
#####  }
#####
#####  output {
#####    #File html_out = "checkout_dir/main.html"
#####    #Array[File] results = glob("checkout_dir/results/*")
#####    File outputs = "inputs"
#####  }
#####
######  runtime {
######    docker: "us.gcr.io/broad-dsde-methods/pyro:1.2.1"
######    bootDiskSizeGb: 50
######    memory: "15G"
######    cpu: 4
######    zones: "us-east1-d us-east1-c us-central1-a us-central1-c us-west1-b"
######    gpuCount: 1
######    gpuType: "nvidia-tesla-k80"
######    maxRetries: 0
######  }
#####}
#####
#####
#####workflow run_ml_with_wdl {
#####
#####  input {
#####     File json_inputs
######     File train_data
######     File test_data
######     File checkpoint
######     String notebook_name
######     String git_repo
######     String commit_or_branch
######     String bucket_output
#####  }
#####
#####  call run_jupyter_notebook {
#####    input :
#####        json_inputs = json_inputs
######        train_data = train_data,
######        test_data = test_data,
######        checkpoint = checkpoint,
######        notebook_name = notebook_name,
######        git_repo = git_repo,
######        commit_or_branch = commit_or_branch,
######        bucket_output = bucket_output
#####  }
#####
#####  output {
#####    File html_out = run_jupyter_notebook.html_out
#####    Array[File] results_out = run_jupyter_notebook.results
#####  }
#####}
