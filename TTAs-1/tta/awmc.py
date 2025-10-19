import os
import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
import json
import torch
from systems.GradientBasedSystem import GradientBasedSystem
from .base import IStrategy

class AWMCStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.tta_config = config["tta_config"]
        self.anchor = GradientBasedSystem(config["system_config"])
        self.leader = GradientBasedSystem(config["system_config"])
        self.system = GradientBasedSystem(config["system_config"])  # chaser
        self.system.eval()
        self.system.snapshot("init")

        # setup anchor and leader
        self.anchor.eval()
        self.leader.eval()
        for param in self.anchor.model.parameters():
            param.detach_()
        for param in self.leader.model.parameters():
            param.detach_()

        self.ema_task_vector = None
        self.alpha = 0.999
        self.opt_param_names = self.anchor.opt_param_names

        # log
        self.transcriptions = []

    @torch.no_grad()
    def _update_leader(self):
        origin_model_state = self.system.history["init"][0]
        task_vector = self._get_task_vector(leader=False)

        if self.ema_task_vector is None:
            self.ema_task_vector = {}
            for name in self.opt_param_names:
                self.ema_task_vector[name] = (1 - self.alpha) * task_vector[name]
        else:
            for name in self.opt_param_names:
                self.ema_task_vector[name] = self.alpha * self.ema_task_vector[name] + (1 - self.alpha) * task_vector[name]
        
        # add back to origin model
        merged_model_state = {}
        for name in origin_model_state:
            if name in self.opt_param_names:
                merged_model_state[name] = origin_model_state[name] + self.ema_task_vector[name]
            else:
                merged_model_state[name] = origin_model_state[name]

        self.leader.history["merged"] = (merged_model_state, None, None)
        self.leader.load_snapshot("merged")

    @torch.no_grad()
    def _get_task_vector(self, leader=False):
        if leader:
            model_state = self.leader.model.state_dict()
        else:
            model_state = self.system.model.state_dict()
        origin_model_state = self.system.history["init"][0]
        task_vector = {
            name: model_state[name] - origin_model_state[name]
        for name in self.opt_param_names}
        return task_vector
    
    def _update(self, wavs): 
        anchor_pl_target = self.anchor.inference(wavs)[0]
        is_collapse = False
        for _ in range(self.tta_config["steps"]):
            leader_pl_target = self.leader.inference(wavs)[0]
            record = {}
            self.system.ctc_adapt_auto(
                wavs=[wavs[0], wavs[0]],
                texts=[anchor_pl_target, leader_pl_target],
                batch_size=1,
                record=record
            )
            if record.get("collapse", False):
                is_collapse = True
            self._update_leader()

        if is_collapse:
            print("oh no")
    
    def run(self, wavs):
        
        self._update(wavs)
        self.system.eval()
        trans = self.system.inference(wavs)[0]

        return trans
    
    def get_adapt_count(self):
        return self.system.adapt_count

    def load_checkpoint(self, path):
        self.system.load(path)
