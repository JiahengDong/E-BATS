from torch.utils.data import Dataset
import yaml
from systems.GradientBasedSystem import GradientBasedSystem
from .base import IStrategy
import torch

class SGEMStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.tta_config = config["tta_config"]
        config["system_config"]["lr"] = config["tta_config"]["lr"]
        config["system_config"]["lm_coef"] = config["tta_config"]["lm_coef"]
        config["system_config"]["beam_width"] = config["tta_config"]["beam_width"]
        config["system_config"]["renyi_entropy_alpha"] = config["tta_config"]["renyi_entropy_alpha"]
        config["system_config"]["negative_sampling_method"] = config["tta_config"]["negative_sampling_method"]
        config["system_config"]["ns_coef"] = config["tta_config"]["ns_coef"]
        config["system_config"]["ns_threshold"] = config["tta_config"]["ns_threshold"]
        #only update feature extractor layers
        config["system_config"]["train_LN"] = False
        config["system_config"]["scheduler"] = config["tta_config"]["scheduler"]
        
        self.system = GradientBasedSystem(config["system_config"])
        self.system.eval()
        self.system.snapshot("init")
        
        
    def _init_start(self) -> None:
        self.system.load_snapshot("init")
    
    def _adapt(self, wavs):
        self.system.eval()
        for _ in range(self.tta_config["steps"]):
            record = {}
            self.system.sgem_adapt(
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
            