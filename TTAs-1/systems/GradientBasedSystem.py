import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from transformers import HubertForCTC, Data2VecAudioForCTC
from copy import deepcopy
import json
import transformers
from utils.tool import batchify
from .loss import softmax_entropy, mcc_loss, div_loss, tc_reg_loss, renyi_entropy
import torch.nn.functional as F
from utils.sgem_forward import get_logits_and_pseudo_labels
from torch.optim.lr_scheduler import CosineAnnealingLR

class GradientBasedSystem(object):

    SAMPLE_RATE = 16000

    def __init__(self, config) -> None:
        self.config = config
        self.history = {}
        self.adapt_count = 0
        self.filterout = 0
        
        # load model and tokenizer
        self.processor = Wav2Vec2Processor.from_pretrained(config["model_name"], sampling_rate=GradientBasedSystem.SAMPLE_RATE, return_attention_mask=True)
        
        if config["model_name"] == "facebook/wav2vec2-base-960h":
            self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        elif config["model_name"] == "facebook/hubert-large-ls960-ft":
            self.model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        
        self.model.eval().cuda()
        
        # set up
        self.model.requires_grad_(False)
        if config["requires_bp"]:
            if config["tta_name"] == "cea":
                params1, param_names1 = self.collect_params()
                self.config["train_feature"] = False
                params2, param_names2 = self.collect_params()
                params = [params1, params2]
                lrs = [config["lr1"], config["lr2"]]
                self.optimizer, self.scheduler = setup_optimizer(params, config["opt"], lrs, scheduler=config["scheduler"])
            else:
                params, self.opt_param_names = self.collect_params()
                self.optimizer, self.scheduler = setup_optimizer(params, config["opt"], config["lr"], scheduler=config["scheduler"])
            if config["tta_name"] == "sgem":
                self.decode_processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

        f = open('vocab.json')
        self.vocab = json.load(f)


    def eval(self):
        self.model.eval()

    def _wav_to_model_input(self, wavs):
        
        inputs = self.processor(
            audio=wavs,
            sampling_rate=GradientBasedSystem.SAMPLE_RATE,
            return_tensors="pt",
            padding="longest",
        )
        
        return inputs.to(device=self.model.device)
    
    def _text_to_model_input(self, texts):
        # target texts need to match wav2vec2's format to make sure correct tokenization
        texts_new = []
        for x in texts:
            x = x.upper()
            x_new = ""
            for s in x:
                if s in self.vocab or s == ' ':
                    x_new += s
            texts_new.append(x_new)

        labels = self.processor(
            text=texts_new,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True
        )

        labels = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        return labels.to(device=self.model.device)

    def reset_adapt_counter(self):
        self.adapt_count = 0
    
    def l2_loss(self):
        l2_loss = 0.0
        orig_state_dict = self.history["init"][0]

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                l2_loss += torch.sum((param - orig_state_dict[name]) ** 2)
        return l2_loss

    def update_model_probs(self,current_model_probs, new_probs):
        if current_model_probs is None:
            if new_probs.size(0) == 0:
                return None
            else:
                with torch.no_grad():
                    return new_probs.mean(0)
        else:
            if new_probs.size(0) == 0:
                with torch.no_grad():
                    return current_model_probs
            else:
                with torch.no_grad():
                    return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
    def tent_adapt(self, wavs):
        inputs = self._wav_to_model_input(wavs)
        inputs = inputs.to(device=self.model.device)
        outputs = self.model(**inputs).logits
        predicted_ids = torch.argmax(outputs, dim=-1)

        if self.config["non_blank"]:
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
            e_loss = softmax_entropy(outputs / self.config["temp"])[non_blank].mean(0).mean()
        else:
            e_loss = softmax_entropy(outputs / self.config["temp"]).mean(0).mean()
        
        e_loss.backward()
        self.optimizer.step()
        self.model.zero_grad()
        
        
    def suta_adapt(self, wavs, record={}):
        """
        Single gradient step on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        the index of <pad> in vocab is 0
        Due to wav2vec2-base special design, attention mask is always none, so ctc input length is always the
        full length, model should learn to output id=0 on the padded part of wav.
        """
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        inputs = inputs.to(device=self.model.device)
        outputs = self.model(**inputs).logits
        predicted_ids = torch.argmax(outputs, dim=-1)

        loss = 0
        if self.config["em_coef"] > 0:
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
            if self.config["non_blank"]:
                e_loss = softmax_entropy(outputs / self.config["temp"])[non_blank].mean(0).mean()
            else:
                e_loss = softmax_entropy(outputs / self.config["temp"]).mean(0).mean()
            loss += e_loss * self.config["em_coef"]
            record["e_loss"] = e_loss.item()
        
        if 1 - self.config["em_coef"] > 0: 
            c_loss = mcc_loss(outputs / self.config["temp"], self.config["reweight"])
            loss += c_loss * (1 - self.config["em_coef"])
            record["c_loss"] = c_loss.item()

        if self.config["div_coef"] > 0: 
            d_loss = div_loss(outputs, self.config["non_blank"]) 
            loss += d_loss * self.config["div_coef"]
            record["d_loss"] = d_loss.item()
        
        record["total_loss"] = loss.item()

        if self.config["l2_coef"] > 0: 
            l2_loss = self.l2_loss()
            loss += l2_loss * self.config["l2_coef"]
            record["l2_loss"] = l2_loss.item()

        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.model.zero_grad()

    def cea_adapt(self, wavs, step, record={}):
        inputs = self._wav_to_model_input(wavs)
        inputs = inputs.to(device=self.model.device)
        loss = 0
        blank_id = 0
        if step < self.config["step1"]:
            outputs = self.model(**inputs).logits
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != blank_id, 1, 0).bool()
            if self.config["em_coef"] > 0:
                e = softmax_entropy(outputs / self.config["temp"])
                e_non_blank = e[non_blank]
                weight = 1/(1+torch.exp(-e_non_blank))
                e_loss = (weight*e_non_blank).mean()
                loss += e_loss * self.config["em_coef"]
                record["e_loss"] = e_loss.item()
            
            if 1 - self.config["em_coef"] > 0:
                c_loss = mcc_loss(outputs / self.config["temp"], self.config["reweight"])
                loss += c_loss * (1 - self.config["em_coef"])
                record["c_loss"] = c_loss.item()
            
            self.model.zero_grad()
            loss.backward()
            self.optimizer[0].step()

        else:
            outputs = self.model(**inputs).logits
            if 'wav2vec2' in self.config["model_name"]:
                feats = self.model.wav2vec2.feature_extractor(**inputs)
            elif 'hubert' in self.config["model_name"]:
                inputs_values = inputs.input_values
                feats = self.model.hubert.feature_extractor(inputs_values)
            
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != blank_id, 1, 0).bool()

            if self.config["em_coef"] > 0:
                e_loss = softmax_entropy(outputs / self.config["temp"]).mean(0).mean()
                loss += e_loss * self.config["em_coef"]
                record["e_loss"] = e_loss.item()

            if 1 - self.config["em_coef"] > 0:
                c_loss = mcc_loss(outputs / self.config["temp"], self.config["reweight"])
                loss += c_loss * (1 - self.config["em_coef"])
                record["c_loss"] = c_loss.item()
            
            tc_loss = tc_reg_loss(feats, non_blank)
            loss += self.config["tc_coef"]*tc_loss

            self.model.zero_grad()
            loss.backward()
            self.optimizer[1].step()

    def sgem_adapt(self, wavs, record={}):
        self.optimizer.zero_grad()
        inputs = self._wav_to_model_input(wavs).to(device=self.model.device)
        outputs, pseudo_labels = get_logits_and_pseudo_labels(self.config, self.model, self.decode_processor, inputs.input_values)   
        
        #beam search with negative sampling
        criterion = nn.CrossEntropyLoss(ignore_index=0) if self.config["non_blank"] else nn.CrossEntropyLoss()
        negative_outputs = outputs.clone()
        negative_loss = 0
        char_history = pseudo_labels[0].to('cuda')
        #use ns3l as negative sampling method as original SGEM setting
        negative_mask = torch.where(torch.softmax(negative_outputs, dim=-1) < self.config["ns_threshold"] * (10 / negative_outputs.shape[-1]), 1, 0)
        negative_loss += torch.mean(-torch.log(1 - torch.sum(negative_mask * torch.softmax(negative_outputs / self.config["temp"], dim=-1), dim=-1)))
        if torch.is_tensor(negative_loss):
            (self.config["ns_coef"] * negative_loss).backward(retain_graph=True)
            
        #calculate renyi_em loss
        predicted_ids = torch.argmax(outputs, dim=-1)
        non_blank = torch.where(predicted_ids != 0, 1, 0).bool()

        if self.config["non_blank"]:
            e_loss = renyi_entropy((outputs / self.config["temp"])[non_blank], alpha=self.config["renyi_entropy_alpha"])
        else:
            e_loss = renyi_entropy(outputs / self.config["temp"], alpha=self.config["renyi_entropy_alpha"])
        e_loss.backward(retain_graph=True)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def eata_adapt(self, wavs, fishers, fisher_alpha, e_margin, d_margin, current_model_probs, num_samples_update):
        """Forward and adapt model on ASR data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        inputs = self._wav_to_model_input(wavs)
        inputs_values = inputs.input_values.to(device=self.model.device)
        outputs = self.model(inputs_values).logits
        predicted_ids = torch.argmax(outputs, dim=-1)

        if self.config["non_blank"]:
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool() 
        else:
            non_blank = torch.ones_like(predicted_ids).bool()
        
        # Calculate entropy only for non-blank tokens
        entropys = softmax_entropy(outputs) 
        entropys = entropys * non_blank.float()  
        entropys = entropys.sum(dim=1) / non_blank.sum(dim=1).clamp(min=1)  
        # Filter unreliable samples
        filter_ids_1 = torch.where(entropys < e_margin) 
        entropys = entropys[filter_ids_1]
        ids2 = torch.where(filter_ids_1[0]>-0.1)
        # Filter redundant samples
        if len(filter_ids_1[0]) == 0:
            updated_probs = current_model_probs
        else:
            if current_model_probs is not None:
                sentence_level_outputs = outputs[filter_ids_1].softmax(dim=-1).mean(dim=1)
                cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), sentence_level_outputs, dim=-1)
                filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
                ids2 = filter_ids_2
                entropys = entropys[filter_ids_2]
                new_probs = outputs[filter_ids_1][filter_ids_2].softmax(dim=-1).mean(dim=1)
            
                updated_probs = self.update_model_probs(current_model_probs, new_probs)

            else:
                new_probs = outputs[filter_ids_1].softmax(dim=-1).mean(dim=1)
                updated_probs = self.update_model_probs(current_model_probs, new_probs)
        
        loss = entropys.mean(0)
        print(loss)
        # Add Fisher regularization if applicable
        if fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in fishers:
                    ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
            loss += ewc_loss
        # Backpropagation and optimization
        if inputs_values[filter_ids_1[0]][ids2[0]].size(0) != 0:
            loss.backward()
            self.optimizer.step()
        else:
            self.filterout += 1
        self.optimizer.zero_grad()
        print("utterances have been filtered out: ", self.filterout)
        return entropys.size(0), filter_ids_1[0].size(0), updated_probs

    def suta_adapt_loss_only(self, wavs, record={}):
        """
        suta_adapt without gradient control so that we can use gradient accumulation
        """
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        # print(type(inputs))  # inputs belongs to a custom dict class defined in transformers, not tensor
        inputs = inputs.to(device=self.model.device)
        outputs = self.model(**inputs)
        predicted_ids = torch.argmax(outputs.logits, dim=-1)

        loss = 0
        if self.config["em_coef"] > 0:
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
            x = softmax_entropy(outputs.logits / self.config["temp"])
            if self.config["non_blank"]:
                x = x[non_blank]
            if len(x) > 0:
                e_loss = x.mean(0).mean()
            else:
                e_loss = torch.tensor(0, device=self.model.device)
                record["collapse"] = True
            loss += e_loss * self.config["em_coef"]
            record["e_loss"] = e_loss.item()
        
        if 1 - self.config["em_coef"] > 0: 
            c_loss = mcc_loss(outputs.logits / self.config["temp"], self.config["reweight"])
            loss += c_loss * (1 - self.config["em_coef"])
            record["c_loss"] = c_loss.item()

        if self.config["div_coef"] > 0: 
            d_loss = div_loss(outputs.logits, self.config["non_blank"]) 
            loss += d_loss * self.config["div_coef"]
            record["d_loss"] = d_loss.item()
        
        record["total_loss"] = loss.item()

        if self.config["l2_coef"] > 0: 
            l2_loss = self.l2_loss()
            loss += l2_loss * self.config["l2_coef"]
            record["l2_loss"] = l2_loss.item()

        return loss

    def suta_adapt_auto(self, wavs, batch_size=-1, record={}) -> None:
        """ suta_adapt auto split to smaller batch """
        self.adapt_count += 1
        if batch_size == -1:
            batch_size == len(wavs)
        self.model.zero_grad()
        denom_scale = len(wavs) // batch_size
        assert denom_scale > 0
        for wavs in batchify(wavs, batch_size=batch_size):
            loss = self.suta_adapt_loss_only(wavs, record=record)
            self.adapt_count -= 1 
            loss = loss / denom_scale
            loss.backward()
    
        self.optimizer.step()
        self.model.zero_grad()

    def ctc_adapt(self, wavs, texts, record={}):
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        labels = self._text_to_model_input(texts)
        if labels.shape[1] == 0:  
            labels = torch.zeros((len(labels), 1), device=labels.device)
            record["collapse"] = True
        inputs["labels"] = labels

        outputs = self.model(**inputs)
        loss = outputs.loss
        record["ctc_loss"] = loss.item()
        record["total_loss"] = loss.item()

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None: 
            self.scheduler.step()
        self.model.zero_grad()

    def ctc_adapt_loss_only(self, wavs, texts, record={}):
        """
        ctc_adapt without gradient control so that we can use gradient accumulation
        """
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        labels = self._text_to_model_input(texts)
        if labels.shape[1] == 0:  
            labels = torch.zeros((len(labels), 1), device=labels.device)
            record["collapse"] = True
    
        outputs = self.model(**inputs)
        logits = outputs.logits 
    
        # Calculate input lengths (needed for CTC loss)
        input_lengths = torch.full(
            (logits.shape[0],), 
            logits.shape[1], 
            dtype=torch.long, 
            device=logits.device
        )
    
        # Calculate target lengths (needed for CTC loss)
        target_lengths = torch.sum(labels != -100, dim=1).to(device=logits.device)
    
        # Replace -100 padding with 0 (blank token) for CTC loss calculation
        labels_for_ctc = labels.clone()
        labels_for_ctc[labels_for_ctc == -100] = 0
        
        # Calculate CTC loss
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = torch.nn.functional.ctc_loss(
            log_probs.transpose(0, 1), 
            labels_for_ctc,
            input_lengths,
            target_lengths,
            blank=0,  
            reduction='mean',
            zero_infinity=True 
        )
        
        record["ctc_loss"] = loss.item()
        record["total_loss"] = loss.item()

        return loss

    def ctc_adapt_auto(self, wavs, texts, batch_size=-1, record={}) -> None:
        """ ctc_adapt auto split to smaller batch """
        self.adapt_count += 1
        if batch_size == -1:
            batch_size == len(wavs)
        self.model.zero_grad()
        total_loss = torch.tensor(0.0, device=self.model.device)
        count = 0
        for wavs, texts in zip(batchify(wavs, batch_size=batch_size), batchify(texts, batch_size=batch_size)):
            loss = self.ctc_adapt_loss_only(wavs, texts, record=record)
            self.adapt_count -= 1 
            total_loss += loss
            count += 1
            print(count)
        total_loss.backward()
    
        self.optimizer.step()
        self.model.zero_grad()

    @torch.no_grad()
    def inference(self, wavs):
        inputs = self._wav_to_model_input(wavs)
        outputs = self.model(**inputs).logits
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        
        return list(transcription)

    @torch.no_grad()
    def calc_suta_loss(self, wavs):
        record = {}
        self.suta_adapt_loss_only(wavs, record=record)
        self.adapt_count -= 1
        return record

    @torch.no_grad()
    def calc_ctc_loss(self, wavs, texts):
        record = {}
        self.ctc_adapt_loss_only(wavs, texts, record=record)
        self.adapt_count -= 1
        return record

    def snapshot(self, key: str):
        """Copy the model and optimizer states for resetting after adaptation."""
        # print(f"Store state. (key: {key})")
        model_state = deepcopy(self.model.state_dict())
        if type(self.optimizer) == list:
            optimizer_state = [deepcopy(optimizer.state_dict()) for optimizer in self.optimizer]
        else:
            optimizer_state = deepcopy(self.optimizer.state_dict())

        if self.scheduler is not None:
            scheduler_state = deepcopy(self.scheduler.state_dict())
        else:
            scheduler_state = None
        self.history[key] = (model_state, optimizer_state, scheduler_state)
    
    def load_snapshot(self, key: str) -> None:
        """Restore the model and optimizer states from copies."""
        # print(f"Reset. (key: {key})")
        model_state, optimizer_state, scheduler_state = self.history[key]
        model_state = deepcopy(model_state)
        self.model.load_state_dict(model_state, strict=True)
        
        if optimizer_state is not None:
            # optimizer_state = self.history["init"][1]
            if type(optimizer_state) == list:
                for optimizer, optimizer_state in zip(self.optimizer, optimizer_state):
                    optimizer.load_state_dict(optimizer_state)
            else:
                optimizer_state = deepcopy(optimizer_state)
                self.optimizer.load_state_dict(optimizer_state)

        if scheduler_state is not None:
            scheduler_state = deepcopy(scheduler_state)
            self.scheduler.load_state_dict(scheduler_state)

    def delete_snapshot(self, key: str) -> None:
        """Delete specific history."""
        self.history.pop(key)

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
    
    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        trainable = []
        if self.config["bias_only"]:
            trainable = ['bias']
        else: 
            trainable = ['weight', 'bias']

        if self.config.get("bitfit", False):
            print("bitfit")
            for np, p in self.model.named_parameters():
                if str(np).split('.')[1] == 'encoder' and "bias" in np:
                    p.requires_grad = True
                    params.append(p)
                    names.append(np)
        
        for nm, m in self.model.named_modules():
            if self.config["train_LN"]: 
                if isinstance(m, nn.LayerNorm):
                    for np, p in m.named_parameters():
                        if np in trainable:
                            p.requires_grad = True
                            params.append(p)
                            names.append(f"{nm}.{np}")

            if self.config["train_feature"]:
                if len(str(nm).split('.')) > 1:
                    if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
                        for np, p in m.named_parameters():
                            p.requires_grad = True
                            params.append(p)
                            names.append(f"{nm}.{np}")
                            
            if self.config["train_all"]: 
                for np, p in m.named_parameters():
                    p.requires_grad = True
                    params.append(p)
                    names.append(f"{nm}.{np}")

        return params, names

def setup_optimizer(params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, step_size=1, gamma=0.7):
    if opt_name == 'Adam':       
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            betas=(beta, 0.999),
            weight_decay=weight_decay,
            eps=1e-8,
            foreach=False,  # Disable foreach implementation which can be non-deterministic
        )
    elif opt_name == 'AdamW':  # AdamW
        if type(lr) != list:
            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                foreach=False,  # For reproducibility #Set to None for SGEM, since it has large performance difference (others have tiny variants)
                weight_decay=weight_decay)
            
        else:
            opt = getattr(torch.optim, opt_name)
            optimizer = [opt(p, lr=l, betas=(beta, 0.999), weight_decay=weight_decay, eps=1e-8, foreach=False, amsgrad=False) for p, l in zip(params, lr)]
    else:
        print("SGD optimizer")
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, foreach=False)

    if scheduler is not None:
        torch.manual_seed(42)
        return optimizer, eval(scheduler)(optimizer, T_max=10, eta_min=0.00002)
    
    return optimizer, None