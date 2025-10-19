from torch.utils.data import Dataset
import yaml
from tqdm import tqdm

from systems.GradientBasedSystem import GradientBasedSystem
from .base import IStrategy
import torch

class CEAStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.tta_config = config["tta_config"]
        config["system_config"]["lr1"] = config["tta_config"]["lr1"]
        config["system_config"]["lr2"] = config["tta_config"]["lr2"]
        config["system_config"]["tc_coef"] = config["tta_config"]["tc_coef"]
        config["system_config"]["step1"] = config["tta_config"]["step1"]
        config["system_config"]["step2"] = config["tta_config"]["step2"]
        self.system = GradientBasedSystem(config["system_config"])
        self.system.eval()
        self.system.snapshot("init")

    def _init_start(self) -> None:
        self.system.load_snapshot("init")
    
    def _adapt(self, wavs):
        self.system.eval()
        for i in range(self.tta_config["step1"] + self.tta_config["step2"]):
            record = {}
            self.system.cea_adapt(
                wavs=wavs,
                step=i,
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
