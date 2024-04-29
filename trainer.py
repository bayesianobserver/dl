"""
Trainer code for any nn model. 
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from utils import CfgNode as CN
import numpy as np

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()

        # device to train on
        C.device = 'auto'
        
        # dataloder parameters
        C.num_workers = 4 
        
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 100
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95) # this is meant for AdamW
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.eval_iters = 500
        return C

    def __init__(self, config, model, train_dataset, test_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.callbacks = defaultdict(list)
        self.eval_iters = config.eval_iters

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)



    def run(self):
        model, config = self.model, self.config

        # setup the optimizer (basically separates params that don't need weight decay from those that do)
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=True
        )

        test_loader = DataLoader(
            self.test_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=True  
        )

        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        self.train_losses = []
        self.test_losses = []
        while True:

            model.train()
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            #self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # Compuse loss on test dataset and train dataset, every self.eval_iters
            if self.iter_num % self.eval_iters == 0:
                model.eval()
                test_losses = []
                train_losses = []
                with torch.inference_mode():
                    for b, (x_test_batch, y_test_batch) in enumerate(test_loader):
                        test_preds, test_loss = model(x_test_batch, y_test_batch)
                        test_losses.append(test_loss)
                    for b, (x_train_batch, y_train_batch) in enumerate(train_loader):
                        train_preds, train_loss = model(x_train_batch, y_train_batch)
                        train_losses.append(train_loss)
                print("iter_num", self.iter_num, " train_loss:", np.mean(train_losses), ", test_loss: ", np.mean(test_losses))
                self.test_losses.append([self.iter_num, np.mean(test_losses)])
                self.train_losses.append([self.iter_num, np.mean(train_losses)])

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
