version 1.0

task run_jupyter_notebook {
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
     echo $PWD
     echo $(ls)

     git clone ~{git_repo} checkout_dir
     cd checkout_dir
     git checkout ~{commit_or_branch}

     echo $PWD
     echo $(ls)

     #run the notebook
  }

  output {
    String str_out = read_string(stdout())
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
}


workflow run_ml_with_wdl {
  
  input {
     File train_data
     File test_data
     File checkpoint
     String notebook_name
     String git_repo
     String commit_or_branch
     String bucket_output
  }
  
  call run_jupyter_notebook {
    input :
        train_data = train_data,
        test_data = test_data,
        checkpoint = checkpoint,
        notebook_name = notebook_name,
        git_repo = git_repo,
        commit_or_branch = commit_or_branch,
        bucket_output = bucket_output
  }

  output {
    String str_out = run_jupyter_notebook.str_out

  }
}
