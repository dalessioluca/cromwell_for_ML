version 1.0

task run_jupyter_notebook {
  input {
        File train_data
        File test_data
        File checkpoint

        String notebook_name = "main.ipynb"
        String git_repo
        String commit_or_branch
        String bucket_output
    }
  
  command {

     # checkout the repo and the commit you want
     set -e
     git clone ~{git_repo} checkout
     cd checkout
     git checkout ~{commit_or_branch}

     #run the notebook
     #pip install matplotlib
     #pip install jupyter_contrib_nbextensions
     #pip install moviepy
     #jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to=html --execute ~{notebook_name} --output main_output.html
     jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to=html --execute ~{notebook_name} --output main_output.html

  }
  output {
    File main_output_html = "main_output.html"
    Array[File] results = glob("results/*")
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
    File html_out = run_jupyter_notebook.main_output_html
    Array[File] results_out = run_jupyter_notebook.results
  }
}
