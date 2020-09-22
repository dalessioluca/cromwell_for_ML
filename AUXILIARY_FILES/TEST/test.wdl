version 1.0

task test {
    input {
        String neptune_api_token 
        String neptune_project 
    }

    command <<<
        pip install neptune-client
        export NEPTUNE_API_TOKEN="~{neptune_api_token}"

        python <<<CODE
        
        import neptune
        
        NEPTUNE_PROJECT=~{neptune_project}
        print(NEPTUNE_PROJECT)
#        neptune.set_project("NEPTUNE_PROJECT")
#        exp = neptune.create_experiment()
# 
#        x = 1.0
#        for n in range(100):
#            exp.log_metric('x', x)
#            x *= 0.9
# 
#        exp.stop()
        CODE
    >>>
    
    runtime {
          docker: "python"
          cpu: 1
          preemptible: 3
    }
    
}

workflow test_neptune {

    input {
        String neptune_api_token 
        String neptune_project 
    }

    call test { 
        input :
            neptune_api_token = neptune_api_token, 
            neptune_project = neptune_project
    }
}
