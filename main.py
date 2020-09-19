#!/usr/bin/env python
# coding: utf-8

# # VAE IN PYTORCH RUNNING IN CROMWELL CONNECTED TO NEPTUNE

# In this example I log in neptune:
# 1. images
# 2. model architecture
# 3. model weights
# 4. numerical metrics 
# 5. source code

import matplotlib.pyplot as plt
import torch
import numpy
import neptune
from MODULES.utilities import *
from MODULES.vae_model import * 
from MODULES.utilities_neptune import *

# read the parameters
params = load_json_as_dict("./ML_parameters.json")

# create the neptune experiment
neptune.set_project(params["neptune_project"])

exp = neptune.create_experiment(params=flatten_dict(params),
                                upload_source_files=["./MODULES/vae_model.py"])

# create the dataset and visualize them
BATCH_SIZE = params["simulation"]["BATCH_SIZE"]

train_loader = SpecialDataSet(img=load_obj("data_train.pt"),
                              store_in_cuda=torch.cuda.is_available(),
                              shuffle=True,
                              drop_last=True,
                              batch_size=BATCH_SIZE)

test_loader = SpecialDataSet(img=load_obj("data_test.pt"),
                             store_in_cuda=torch.cuda.is_available(),
                             shuffle=False,
                             drop_last=True,
                             batch_size=BATCH_SIZE)

train_batch_example = train_loader.check_batch()
exp.log_image("train_batch_example", train_batch_example)
#train_batch_example

test_batch_example = test_loader.check_batch()
exp.log_image("test_batch_example", test_batch_example)
#test_batch_example

reference_imgs, labels, index = test_loader.load(8)
tmp = show_batch(reference_imgs, n_padding=4, figsize=(12,12), title='reference imgs')
exp.log_image("reference_imgs", tmp)
#tmp

# Initialize model and optimizer
vae = SimpleVae(params)
log_model_summary(experiment=exp, model=vae)
optimizer = instantiate_optimizer(model=vae, dict_params_optimizer=params["optimizer"])

# instantiate the scheduler if necessary    
if params["optimizer"]["scheduler_is_active"]:
    scheduler = instantiate_scheduler(optimizer=optimizer, dict_params_scheduler=params["optimizer"])

# The train loop

TEST_FREQUENCY = params["simulation"]["TEST_FREQUENCY"]
CHECKPOINT_FREQUENCY = params["simulation"]["CHECKPOINT_FREQUENCY"]
NUM_EPOCHS = params["simulation"]["MAX_EPOCHS"]
min_test_loss = 999999

for epoch in range(NUM_EPOCHS):
        
    with torch.enable_grad():
        vae.train()
        train_metrics = process_one_epoch(model=vae, 
                                          dataloader=train_loader, 
                                          optimizer=optimizer, 
                                          verbose=(epoch==0),
                                          weight_clipper=None)
             
        print("Train "+train_metrics.pretty_print(epoch))
        log_metrics(exp, train_metrics, prefix="train_")
    
    if(epoch % TEST_FREQUENCY == 0):
        with torch.no_grad():
            vae.eval()
            test_metrics = process_one_epoch(model=vae, 
                                             dataloader=test_loader, 
                                             optimizer=optimizer, 
                                             verbose=(epoch==0), 
                                             weight_clipper=None)
            
            print("Test  "+test_metrics.pretty_print(epoch))
            log_metrics(exp, test_metrics, prefix="test_")
        
            min_test_loss = min(min_test_loss, test_metrics.loss)
            
            imgs_rec = vae.forward(imgs_in=reference_imgs).imgs
            tmp = show_batch(imgs_rec, n_padding=4, figsize=(12,12), title='epoch= {0:6d}'.format(epoch))
            exp.log_image("imgs_rec", tmp)
                        
            if((test_metrics.loss == min_test_loss) or ((epoch % CHECKPOINT_FREQUENCY) == 0)): 
                ckpt = create_ckpt(model=vae, 
                                   optimizer=optimizer, 
                                   epoch=epoch, 
                                   hyperparams_dict=params)
                save_obj(ckpt, "last_ckpt.pt")  # save locally to file 
                log_last_ckpt(exp, "last_ckpt.pt")  # log file into neptune

# sample from the generator
vae.eval()
sample = vae.generate(imgs_in=reference_imgs)
tmp = show_batch(sample, n_padding=4, figsize=(12,12), title='sample from generator')
exp.log_image("sample", tmp)

# terminate the experiment
exp.stop()
