from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
import os
import torch
from systems.GradientBasedSystem import GradientBasedSystem
from .base import IStrategy
import numpy as np

class Buffer(object):
    """ Easy implementation of buffer, use .data to access all data. """

    data: list

    def __init__(self, max_size: int=100) -> None:
        self.max_size = max_size
        self.data = []

    def update(self, x):
        self.data.append(x)
        if len(self.data) > self.max_size:
            self.data.pop(0)
    
    def clear(self):
        self.data.clear()


class DSUTAStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.tta_config = config["tta_config"]  
        self.system = GradientBasedSystem(config["system_config"])
        self.system.eval()

        # Set slow system
        self.slow_system = GradientBasedSystem(config["system_config"])
        self.slow_system.eval()
        self.slow_system.snapshot("start")
        self.timestep = 0
        self.update_freq = config["tta_config"]["update_freq"]
        self.memory = Buffer(max_size=config["tta_config"]["memory"])

        self.system.snapshot("start")

    def _init_start(self):
        self.system.load_snapshot("start")

    def _adapt(self, wavs):
        self.system.eval()
        is_collapse = False
        for _ in range(self.tta_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=wavs,
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")
    
    def _update(self, wavs):
        self.memory.update(wavs[0])
        if (self.timestep + 1) % self.update_freq == 0:
            self.slow_system.load_snapshot("start")
            self.slow_system.eval()
            record = {}
            self.slow_system.suta_adapt_auto(
                wavs=self.memory.data,
                batch_size=1,
                record=record,
            )
            if record.get("collapse", False):
                print("oh no")
            self.slow_system.snapshot("start")
            self.memory.clear()
        self.system.history["start"] = self.slow_system.history["start"]  # fetch start point from slow system
        
    def run(self, wavs):
            
        self._init_start()
        self._adapt(wavs)

        self.system.eval()
        trans = self.system.inference(wavs)[0]

        self._update(wavs)
        self.timestep += 1

        return trans

    
    def get_adapt_count(self):
        return self.system.adapt_count + self.slow_system.adapt_count

    def load_checkpoint(self, path):
        self.system.load(path)
