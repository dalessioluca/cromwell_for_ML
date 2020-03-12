version 1.0

# Note that cromshell will automatically localize file in the following way:
# gs://ld-data-bucket/data/fashionmnist_test.pkl -> /cromwell_roo/data/fashionmnist_test.pkl


# INFO ABOUT ATTACHING DISKS
# https://cloud.google.com/sdk/gcloud/reference/compute/instances/attach-disk


task run_jupyter_disk_mount {
    input {
        File file_train
        File file_test
        String persistent_disk
    }

    parameter_meta {
        file_train: { localization_optional: true }
        file_test: { localization_optional: true }
    }

    command <<<


        echo "start here"
        whoami
        pwd
        ls -lah
        sudo -l

        # get the current zone and vm name
        vm_name=$(curl http://metadata.google.internal/computeMetadata/v1/instance/name -H Metadata-Flavor:Google)
        project_and_zone=$(curl http://metadata.google.internal/computeMetadata/v1/instance/zone -H Metadata-Flavor:Google)
        zone=$(echo $project_and_zone | awk -F'/' '{print $4}' )
        echo $vm_name
        echo $zone

        # Attache the existing disk with a random disk ID
        random_disk_id=$(cat /dev/urandom | env LC_CTYPE=C tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
        gcloud compute instances attach-disk $vm_name --disk ~{persistent_disk} --device-name $random_disk_id --mode ro --zone $zone

        # Mount the disk
        echo "for debug 1"
        lsblk

        # Get internal device name and mount it (assumes that attached divice is the last listed by lsblk)
        internal_device_name=$(lsblk | tail -1 | awk '{print $1}')

        echo "internal_device_name" $internal_device_name
        sudo mkdir -p /cromwell_root/read_only_disk
        sudo ls
        sudo mount -o norecovery,discard,defaults /dev/$internal_device_name /cromwell_root/read_only_disk
        echo "in mounted disk we find"
        ls -lah /cromwell_root/read_only_disk

        # Make links from the mounted disk to the execution directory
        ln -s /cromwell_root/read_only_disk/~{file_train} ./~{file_train}
        ln -s /cromwell_root/read_only_disk/~{file_test} ./~{file_test}
        echo "in execution dir we find"
        ls -lah ./

    >>>

    output {
        File std_out = stdout()
    }

    runtime {
        #docker: "python"
        #docker: "gcr.io/google.com/cloudsdktool/cloud-sdk:latest"  # does not have sudo
        docker: "us.gcr.io/broad-dsde-methods/gsutil_with_sudo:latest"
        zones: "us-east1-d"  # must select the same zone as the persistent-disk I want to attach
        cpu: 1
        preemptible: 3
    }
}


workflow wdl {

    input {
        File file_train
        File file_test
        String persistent_disk
    }

    call run_jupyter_disk_mount {
        input :
            file_train = file_train,
            file_test = file_test,
            persistent_disk = persistent_disk
    }
    output {
        File std_out = run_jupyter_disk_mount.std_out
    }
}






