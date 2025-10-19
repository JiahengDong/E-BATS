from torch.utils.data import Dataset
import yaml
from systems.GradientBasedSystem import GradientBasedSystem
from .base import IStrategy
import torch

class SUTAStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.tta_config = config["tta_config"]
        self.system = GradientBasedSystem(config["system_config"])
        self.system.eval()
        self.system.snapshot("init")
        
    def _init_start(self) -> None:
        self.system.load_snapshot("init")
    
    def _adapt(self, wavs):
        self.system.eval()
        for _ in range(self.tta_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=wavs,
                record=record,
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
            

class CSUTAStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.tta_config = config["tta_config"]
        self.system = GradientBasedSystem(config["system_config"])
        self.system.eval()
    
    def _init_start(self, wavs) -> None:
        pass
    
    def _adapt(self, wavs):
        self.system.eval()
        for _ in range(self.tta_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=wavs,
                record=record,
            )

    def _update(self, wavs):
        pass
    
    def run(self, wavs):
        self._init_start(wavs)
        self._adapt(wavs)
        self.system.eval()
        trans = self.system.inference(wavs)[0]
        self._update(wavs)
        
        return trans

class NoAdaptStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.system = GradientBasedSystem(config["system_config"])
        self.system.eval()
    
    def run(self, wavs):
        return self.system.inference(wavs)[0]
