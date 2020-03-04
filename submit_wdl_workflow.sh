#!/usr/bin/env bash

# This script takes as input:
# 1. the workflow.wdl file
# 2. the input.json

# It does:
# 1. copy input.json to a gs://google_bucket/random_hash.json
# 2. create the tmp.json with the location of gs://google_bucket/random.json to run the wdl workflow
# 3. run cromshel, i.e.:
#    cromshell submit jupyter.wdl wdl_input.json
# 4. copy the input.json into the cromwell-v47.dsde-methods.broadinstitute.org/fd0c78d0-fc67-4c59-af79-e56e0c29b232 together with the wdl file 


# Define some defaults
BUCKET=${BUCKET:-"gs://ld-results-bucket/input_jsons"}
WDL=${WDL:-"jupyter.wdl"}
JSON=${JSON:-"parameters.json"}
SCRIPTNAME=$( echo $0 | sed 's#.*/##g' )


# Helper functions
display_help() {
  echo -e ""
  echo -e "-- $SCRIPTNAME --"
  echo -e "Submit wdl workflow using cromshell"
  echo -e ""
  echo -e " Example usage:"
  echo -e "   $SCRIPTNAME $WDL $JSON $BUCKET"
  echo -e "   $SCRIPTNAME $WDL $JSON"
  echo -e "   $SCRIPTNAME $WDL"
  echo -e "   $SCRIPTNAME"
  echo -e "   $SCRIPTNAME -h"
  echo -e ""
  echo -e ""
  echo -e " Supported Flags:"
  echo -e "   -h                     Display this message"
  echo -e ""
  echo -e " Default behavior:"
  echo -e "   If no inputs are specified the default value will be used"
  echo -e ""
  echo -e ""
}

exit_with_error() {
  echo -e "ERROR!. Something went wrong"
  exit 1
}

exit_with_success() {
  echo -e "GREAT!. Everything went smoothly"
  exit 0
}


# read the input from command line
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h|--help)
    display_help
    exit
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

for i in "${POSITIONAL[@]}"
do
   if [[ $i == *.json ]]; then JSON=$i; fi
   if [[ $i == *.wdl ]]; then WDL=$i; fi
   if [[ $i == gs://* ]]; then BUCKET=$i; fi
done


# At this point I have these trhee values:
echo "Current values: --" $WDL $JSON $BUCKET



# 1. make copy of input file to cloud with random hash
echo
echo "Step1: copying json file into google bucket"
RANDOM_HASH=$(cat /dev/urandom | env LC_CTYPE=C tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
path_to_bucket="${BUCKET}/${RANDOM_HASH}.json"
path_to_bucket_with_double_quotes="\"$path_to_bucket\""
gsutil cp $JSON $path_to_bucket

# 2. create a new wdl_inputs.json
echo
echo "Step2: crerating tmp.json with the inputs for the workflow"
womtool inputs $WDL | awk -v RHS=$path_to_bucket_with_double_quotes '{if(NF>1) print $1 " " RHS ; else print $1}' > tmp.json


# 3. run cromshell
echo
echo "Step3: run cromshell"
cromshell submit $WDL tmp.json | tee tmp_out_from_cromshell  # with "tee" output is both to stdout and to file
rm tmp.json


# 4. copy the parameter file used into the cromshell directory
echo
echo "Step4: copy parameters.json into cromshell directory"
CROMSHELL_CONFIG_DIR=${HOME}/.cromshell
id_just_submitted=$(cat tmp_out_from_cromshell | awk -F"\"" '{print $4}')

last_line_from_table=$( tail -1 "${CROMSHELL_CONFIG_DIR}/all.workflow.database.tsv")
id_last_line=$(echo $last_line_from_table | awk '{print $3}')
general_output_dir=$(echo $last_line_from_table | awk '{print $2}' | sed 's/https:\/\///')

if [[ $id_just_submitted == *$id_last_line* ]]; then 
   # This means that I have grabbed the right line from the table
   output_dir="${CROMSHELL_CONFIG_DIR}/${general_output_dir}/${id_last_line}"
   if [ -d $output_dir ]; then
      # This means that the directory exists
      cp $JSON $output_dir
   fi	
fi

exit_with_success
