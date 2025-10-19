from torch.utils.data import Dataset
import yaml
from systems.GradientBasedSystem import GradientBasedSystem
from .base import IStrategy
import torch

class TNETStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.tta_config = config["tta_config"]
        config["system_config"]["train_feature"] = False
        config["system_config"]["temp"] = 1
        
        self.system = GradientBasedSystem(config["system_config"])
        self.system.eval()
        self.system.snapshot("init")
        
    def _init_start(self) -> None:
        self.system.load_snapshot("init")
    
    def _adapt(self, wavs):
        self.system.eval()
        for _ in range(self.tta_config["steps"]):
            self.system.tent_adapt(
                wavs=wavs, 
            )

    def _update(self, wavs):
        pass
    
    def run(self, wavs):
        self._init_start()
        self._adapt(wavs)
        self.system.eval()
        trans = self.system.inference(wavs)[0]
        self._update(wavs)
        
        return trans
            
