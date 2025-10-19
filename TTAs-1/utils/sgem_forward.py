import math
from typing import Any, Dict, List, Optional, Tuple, Union
import heapq
import numpy as np
import torch
from pyctcdecode.alphabet import BPE_TOKEN
from pyctcdecode.constants import DEFAULT_HOTWORD_WEIGHT, DEFAULT_MIN_TOKEN_LOGP, DEFAULT_PRUNE_BEAMS, DEFAULT_PRUNE_LOGP, MIN_TOKEN_CLIP_P
from pyctcdecode.language_model import HotwordScorer
import kenlm
from systems.loss import log_softmax

# for ctc-based models and conformers
Frames = Tuple[int, int]
WordFrames = Tuple[str, Frames]
LMBeam = Tuple[str, str, str, Optional[str], List[Frames], Frames, float, float]
LMState = Optional[Union["kenlm.State", List["kenlm.State"]]]
OutputBeam = Tuple[str, LMState, List[WordFrames], float, float]
OutputBeamMPSafe = Tuple[str, List[WordFrames], float, float]
NULL_FRAMES: Frames = (-1, -1)  # placeholder that gets replaced with positive integer frame indices


def forward_batch(config, model, processor, wavs, labels=None):
    outputs = forward_ctc_or_conformer(config, model, processor, wavs, labels)
    return outputs

def forward_ctc_or_conformer(config, model, processor, wavs, labels):
    logits = model(wavs).logits
    
    if labels == None or not config["lm_coef"]:
        return logits
    else:
        lm_logits = forward_ctc_or_conformer_with_labels(
            config,
            processor.decoder,
            np.clip(log_softmax(logits.squeeze(0).detach().cpu().numpy(), axis=1),
            np.log(MIN_TOKEN_CLIP_P), 0),
            labels,
            hotword_scorer=HotwordScorer.build_scorer(None, weight=DEFAULT_HOTWORD_WEIGHT),
            lm_start_state=None,
        ).unsqueeze(0)
        return logits + config["lm_coef"] * lm_logits

def forward_ctc_or_conformer_with_labels(
    config,
    model,
    logits,
    labels,
    hotword_scorer,
    lm_start_state,
):
    def _merge_tokens(token_1: str, token_2: str) -> str:
        """Fast, whitespace safe merging of tokens."""
        if len(token_2) == 0:
            text = token_1
        elif len(token_1) == 0:
            text = token_2
        else:
            text = token_1 + " " + token_2
        return text

    def get_new_beams(
        model,
        beams,
        idx_list,
        frame_idx,
        logit_col,
    ):
        new_beams = []
        # bpe we can also have trailing word boundaries ▁⁇▁ so we may need to remember breaks
        force_next_break = False
        for idx_char in idx_list:
            p_char = logit_col[idx_char]
            char = model._idx2vocab[idx_char]
            for (
                text,
                next_word,
                word_part,
                last_char,
                text_frames,
                part_frames,
                logit_score,
                idx_history,
                lm_logits,
            ) in beams:
                if char == "" or last_char == char:
                    if char == "":
                        new_end_frame = part_frames[0]
                    else:
                        new_end_frame = frame_idx + 1
                    new_part_frames = (
                        part_frames if char == "" else (part_frames[0], new_end_frame)
                    )
                    new_beams.append(
                        (
                            text,
                            next_word,
                            word_part,
                            char,
                            text_frames,
                            new_part_frames,
                            logit_score + p_char,
                            idx_history + [idx_char],
                            lm_logits,
                        )
                    )
                # if bpe and leading space char
                elif model._is_bpe and (char[:1] == BPE_TOKEN or force_next_break):
                    force_next_break = False
                    # some tokens are bounded on both sides like ▁⁇▁
                    clean_char = char
                    if char[:1] == BPE_TOKEN:
                        clean_char = clean_char[1:]
                    if char[-1:] == BPE_TOKEN:
                        clean_char = clean_char[:-1]
                        force_next_break = True
                    new_frame_list = (
                        text_frames if word_part == "" else text_frames + [part_frames]
                    )
                    new_beams.append(
                        (
                            text,
                            word_part,
                            clean_char,
                            char,
                            new_frame_list,
                            (frame_idx, frame_idx + 1),
                            logit_score + p_char,
                            idx_history + [idx_char],
                            lm_logits,
                        )
                    )
                # if not bpe and space char
                elif not model._is_bpe and char == " ":
                    new_frame_list = (
                        text_frames if word_part == "" else text_frames + [part_frames]
                    )
                    new_beams.append(
                        (
                            text,
                            word_part,
                            "",
                            char,
                            new_frame_list,
                            NULL_FRAMES,
                            logit_score + p_char,
                            idx_history + [idx_char],
                            lm_logits,
                        )
                    )
                # general update of continuing token without space
                else:
                    new_part_frames = (
                        (frame_idx, frame_idx + 1)
                        if part_frames[0] < 0
                        else (part_frames[0], frame_idx + 1)
                    )
                    new_beams.append(
                        (
                            text,
                            next_word,
                            word_part + char,
                            char,
                            text_frames,
                            new_part_frames,
                            logit_score + p_char,
                            idx_history + [idx_char],
                            lm_logits,
                        )
                    )
        return new_beams

    def get_lm_beams(
        model,
        beams,
        hotword_scorer,
        cached_lm_scores,
        cached_partial_token_scores,
        is_eos,
    ):
        lm_score_list = np.zeros(len(beams))
        language_model = model._language_model
        new_beams = []
        for text, next_word, word_part, last_char, frame_list, frames, logit_score, idx_history, lm_logits in beams:
            new_text = _merge_tokens(text, next_word)
            if new_text not in cached_lm_scores:
                _, prev_raw_lm_score, start_state = cached_lm_scores[text]
                score, end_state = language_model.score(start_state, next_word, is_last_word=is_eos)
                raw_lm_score = prev_raw_lm_score + score
                lm_hw_score = raw_lm_score + hotword_scorer.score(new_text)
                cached_lm_scores[new_text] = (lm_hw_score, raw_lm_score, end_state)
            lm_score, _, _ = cached_lm_scores[new_text]

            if len(word_part) > 0:
                if word_part not in cached_partial_token_scores:
                    # if prefix available in hotword trie use that, otherwise default to char trie
                    if word_part in hotword_scorer:
                        cached_partial_token_scores[word_part] = hotword_scorer.score_partial_token(
                            word_part
                        )
                    else:
                        cached_partial_token_scores[word_part] = language_model.score_partial_token(
                            word_part
                        )
                lm_score += cached_partial_token_scores[word_part]
    
            new_beams.append(
                (
                    new_text,
                    "",
                    word_part,
                    last_char,
                    frame_list,
                    frames,
                    logit_score,
                    logit_score + lm_score,
                    idx_history,
                    lm_logits,
                )
            )
            lm_score_list[model._vocab2idx[last_char]] = lm_score
        
        new_beams_with_lm_logits = []
        for text, next_word, word_part, last_char, frame_list, frames, logit_score, combined_score, idx_history, lm_logits in new_beams:
            new_beams_with_lm_logits.append(
                (
                    text,
                    next_word,
                    word_part,
                    last_char,
                    frame_list,
                    frames,
                    logit_score,
                    combined_score,
                    idx_history,
                    lm_logits + [lm_score_list],
                )
            )
        return new_beams_with_lm_logits

    language_model = model._language_model
    if lm_start_state is None and language_model is not None:
        cached_lm_scores: Dict[str, Tuple[float, float, LMState]] = {
            "": (0.0, 0.0, language_model.get_start_state())
        }
    else:
        cached_lm_scores = {"": (0.0, 0.0, lm_start_state)}
    cached_p_lm_scores: Dict[str, float] = {}
    if not hasattr(model, '_vocab2idx'):
        model._vocab2idx = {vocab: idx for idx, vocab in model._idx2vocab.items()}
    beams = [("", "", "", None, [], NULL_FRAMES, 0.0, [], [])] # start with single beam to expand on

    for frame_idx, logit_col in enumerate(logits):
        idx_list = list(range(0, logit_col.shape[-1]))
        new_beams = get_new_beams(
            model,
            beams,
            idx_list,
            frame_idx,
            logit_col,
        )
        scored_beams = get_lm_beams(
            model,
            new_beams,
            hotword_scorer,
            cached_lm_scores,
            cached_p_lm_scores,
            is_eos=False,
        )
        beams = [scored_beams[labels[frame_idx]][:-3] + scored_beams[labels[frame_idx]][-2:]]
    return torch.tensor(np.array(beams[0][-1])).to('cuda')

@torch.no_grad()
def encode_batch(model, wavs):
    logits = model(wavs).logits
    logitlen = torch.tensor([logits.shape[1]]).to('cuda')
    outputs = logits, logitlen
    return outputs

@torch.no_grad()
def decode_batch(config, model, processor, encoder_output, encoder_length):
    beam_width = config["beam_width"]
    
    pseudo_labels = decode_ctc_or_conformer(
        processor.decoder,
        logits=np.clip(log_softmax(encoder_output.squeeze(0).detach().cpu().numpy(), axis=1), np.log(MIN_TOKEN_CLIP_P), 0),
        beam_width=beam_width,
        beam_prune_logp=DEFAULT_PRUNE_LOGP,
        token_min_logp=DEFAULT_MIN_TOKEN_LOGP,
        prune_history=DEFAULT_PRUNE_BEAMS,
        hotword_scorer=HotwordScorer.build_scorer(None, weight=DEFAULT_HOTWORD_WEIGHT),
        lm_start_state=None,
    )
    return pseudo_labels

def decode_ctc_or_conformer(
    model,
    logits,
    beam_width,
    beam_prune_logp,
    token_min_logp,
    prune_history,
    hotword_scorer,
    lm_start_state,
):
    def _merge_beams(beams):
        """Merge beams with same prefix together."""
        beam_dict = {}
        for text, next_word, word_part, last_char, text_frames, part_frames, logit_score, idx_history in beams:
            new_text = _merge_tokens(text, next_word)
            hash_idx = (new_text, word_part, last_char)
            if hash_idx not in beam_dict:
                beam_dict[hash_idx] = (
                    text,
                    next_word,
                    word_part,
                    last_char,
                    text_frames,
                    part_frames,
                    logit_score,
                    idx_history
                )
            else:
                beam_dict[hash_idx] = (
                    text,
                    next_word,
                    word_part,
                    last_char,
                    text_frames,
                    part_frames,
                    _sum_log_scores(beam_dict[hash_idx][-2], logit_score),
                    idx_history
                )
        return list(beam_dict.values())

    def _sort_and_trim_beams(beams, beam_width: int):
        """Take top N beams by score."""
        return heapq.nlargest(beam_width, beams, key=lambda x: x[-2])

    def _merge_tokens(token_1: str, token_2: str) -> str:
        """Fast, whitespace safe merging of tokens."""
        if len(token_2) == 0:
            text = token_1
        elif len(token_1) == 0:
            text = token_2
        else:
            text = token_1 + " " + token_2
        return text

    def _sum_log_scores(s1: float, s2: float) -> float:
        """Sum log odds in a numerically stable way."""
        # this is slightly faster than using max
        if s1 >= s2:
            log_sum = s1 + math.log(1 + math.exp(s2 - s1))
        else:
            log_sum = s2 + math.log(1 + math.exp(s1 - s2))
        return log_sum

    def get_new_beams(
        model,
        beams,
        idx_list,
        frame_idx,
        logit_col,
    ):
        new_beams = []
        # bpe we can also have trailing word boundaries ▁⁇▁ so we may need to remember breaks
        force_next_break = False

        for idx_char in idx_list:
            p_char = logit_col[idx_char]
            char = model._idx2vocab[idx_char]
            for (
                text,
                next_word,
                word_part,
                last_char,
                text_frames,
                part_frames,
                logit_score,
                idx_history,
            ) in beams:
                if char == "" or last_char == char:
                    if char == "":
                        new_end_frame = part_frames[0]
                    else:
                        new_end_frame = frame_idx + 1
                    new_part_frames = (
                        part_frames if char == "" else (part_frames[0], new_end_frame)
                    )
                    new_beams.append(
                        (
                            text,
                            next_word,
                            word_part,
                            char,
                            text_frames,
                            new_part_frames,
                            logit_score + p_char,
                            idx_history + [idx_char],
                        )
                    )
                # if bpe and leading space char
                elif model._is_bpe and (char[:1] == BPE_TOKEN or force_next_break):
                    force_next_break = False
                    # some tokens are bounded on both sides like ▁⁇▁
                    clean_char = char
                    if char[:1] == BPE_TOKEN:
                        clean_char = clean_char[1:]
                    if char[-1:] == BPE_TOKEN:
                        clean_char = clean_char[:-1]
                        force_next_break = True
                    new_frame_list = (
                        text_frames if word_part == "" else text_frames + [part_frames]
                    )
                    new_beams.append(
                        (
                            text,
                            word_part,
                            clean_char,
                            char,
                            new_frame_list,
                            (frame_idx, frame_idx + 1),
                            logit_score + p_char,
                            idx_history + [idx_char],
                        )
                    )
                # if not bpe and space char
                elif not model._is_bpe and char == " ":
                    new_frame_list = (
                        text_frames if word_part == "" else text_frames + [part_frames]
                    )
                    new_beams.append(
                        (
                            text,
                            word_part,
                            "",
                            char,
                            new_frame_list,
                            NULL_FRAMES,
                            logit_score + p_char,
                            idx_history + [idx_char],
                        )
                    )
                # general update of continuing token without space
                else:
                    new_part_frames = (
                        (frame_idx, frame_idx + 1)
                        if part_frames[0] < 0
                        else (part_frames[0], frame_idx + 1)
                    )
                    new_beams.append(
                        (
                            text,
                            next_word,
                            word_part + char,
                            char,
                            text_frames,
                            new_part_frames,
                            logit_score + p_char,
                            idx_history + [idx_char],
                        )
                    )
        new_beams = _merge_beams(new_beams)
        return new_beams

    def get_lm_beams(
        model,
        beams,
        hotword_scorer: HotwordScorer,
        cached_lm_scores: Dict[str, Tuple[float, float, LMState]],
        cached_partial_token_scores: Dict[str, float],
        is_eos: bool = False,
    ) -> List[LMBeam]:
        """Update score by averaging logit_score and lm_score."""
        # get language model and see if exists
        language_model = model._language_model

        # if no language model available then return raw score + hotwords as lm score
        if language_model is None:
            new_beams = []
            for text, next_word, word_part, last_char, frame_list, frames, logit_score, idx_history in beams:
                new_text = _merge_tokens(text, next_word)
                # note that usually this gets scaled with alpha
                lm_hw_score = (
                    logit_score
                    + hotword_scorer.score(new_text)
                    + hotword_scorer.score_partial_token(word_part)
                )
                new_beams.append(
                    (
                        new_text,
                        "",
                        word_part,
                        last_char,
                        frame_list,
                        frames,
                        logit_score,
                        lm_hw_score,
                        idx_history
                    )
                )
            return new_beams

        new_beams = []
        for text, next_word, word_part, last_char, frame_list, frames, logit_score, idx_history in beams:
            new_text = _merge_tokens(text, next_word)
            if new_text not in cached_lm_scores:
                _, prev_raw_lm_score, start_state = cached_lm_scores[text]
                score, end_state = language_model.score(start_state, next_word, is_last_word=is_eos)
                raw_lm_score = prev_raw_lm_score + score
                lm_hw_score = raw_lm_score + hotword_scorer.score(new_text)
                cached_lm_scores[new_text] = (lm_hw_score, raw_lm_score, end_state)
            lm_score, _, _ = cached_lm_scores[new_text]

            if len(word_part) > 0:
                if word_part not in cached_partial_token_scores:
                    # if prefix available in hotword trie use that, otherwise default to char trie
                    if word_part in hotword_scorer:
                        cached_partial_token_scores[word_part] = hotword_scorer.score_partial_token(
                            word_part
                        )
                    else:
                        cached_partial_token_scores[word_part] = language_model.score_partial_token(
                            word_part
                        )
                lm_score += cached_partial_token_scores[word_part]

            new_beams.append(
                (
                    new_text,
                    "",
                    word_part,
                    last_char,
                    frame_list,
                    frames,
                    logit_score,
                    logit_score + lm_score,
                    idx_history,
                )
            )
        return new_beams

    language_model = model._language_model
    if lm_start_state is None and language_model is not None:
        cached_lm_scores: Dict[str, Tuple[float, float, LMState]] = {
            "": (0.0, 0.0, language_model.get_start_state())
        }
    else:
        cached_lm_scores = {"": (0.0, 0.0, lm_start_state)}
    cached_p_lm_scores: Dict[str, float] = {}
    # start with single beam to expand on
    beams = [("", "", "", None, [], NULL_FRAMES, 0.0, [])]

    for frame_idx, logit_col in enumerate(logits):
        max_idx = logit_col.argmax()
        idx_list = set(np.where(logit_col >= token_min_logp)[0]) | {max_idx}
        new_beams = get_new_beams(
            model,
            beams,
            idx_list,
            frame_idx,
            logit_col,
        )
        # lm scoring and beam pruning
        scored_beams = get_lm_beams(
            model,
            new_beams,
            hotword_scorer,
            cached_lm_scores,
            cached_p_lm_scores,
        )

        # remove beam outliers
        max_score = max([b[-2] for b in scored_beams])
        scored_beams = [b for b in scored_beams if b[-2] >= max_score + beam_prune_logp]
        trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)
        beams = [b[:-2] + (b[-1], ) for b in trimmed_beams]

    new_beams = []
    for text, _, word_part, _, frame_list, frames, logit_score, idx_history in beams:
        new_token_times = frame_list if word_part == "" else frame_list + [frames]
        new_beams.append((text, word_part, "", None, new_token_times, (-1, -1), logit_score, idx_history))
    new_beams = _merge_beams(new_beams)
    scored_beams = get_lm_beams(
        model,
        new_beams,
        hotword_scorer,
        cached_lm_scores,
        cached_p_lm_scores,
        is_eos=True,
    )
    scored_beams = [b[:-2] + (b[-1], ) for b in scored_beams]
    scored_beams = _merge_beams(scored_beams)

    # remove beam outliers
    max_score = max([b[-2] for b in beams])
    scored_beams = [b for b in beams if b[-2] >= max_score + beam_prune_logp]
    trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)

    # remove unnecessary information from beams
    output_beams = [
        torch.tensor(idx_history)
        for _, _, _, _, _, _, _, idx_history in trimmed_beams
    ]
    return output_beams

def get_logits_and_pseudo_labels(config, model, processor, wavs):
     # beam search
    encoder_output, encoder_length = encode_batch(model, wavs)
    pseudo_labels = decode_batch(config, model, processor, encoder_output, encoder_length)
    logits = forward_batch(config, model, processor, wavs, labels=pseudo_labels[0])
    return logits, pseudo_labels