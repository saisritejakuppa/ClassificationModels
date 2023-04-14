import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        #int 16
        self.parser.add_argument('--int16', type=bool, default=False, help='use int16')

        #save the outputs in the folder
        self.parser.add_argument('--save_dir', type=str, default='./outputs', help='path to save the outputs')

        #model
        self.parser.add_argument('--model', type=str, default='resnet18', help='model to use')

        #model_save_path
        self.parser.add_argument('--model_save_path', type=str, default='./model_outputs', help='path to save the model')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt