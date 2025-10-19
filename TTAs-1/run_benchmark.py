import os
import gc
import argparse
import yaml
import json
import pickle
import jiwer  # Import the whole module
from utils.tool import seed_everything
from tta.load import get_tta_cls
from data import load_dataset
from tqdm import tqdm
import torch
import numpy as np

def create_config(args):
    """ Create a dictionary for full configuration """
    res = {
        "tta_name": args.tta_name,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "path": args.path,
        "noise_type": args.noise_type,
        "noise_level": args.noise_level,
        "extra_noise": args.extra_noise,
    }

    system_config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    res["system_config"] = system_config
    res["system_config"]["tta_name"] = args.tta_name
    res["tta_config"] = {}
    for path in args.tta_config:
        tta_config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        res["tta_config"].update(tta_config)

    return res

def main(args):
    config = create_config(args)

    tta = get_tta_cls(args.tta_name)(config)
    dataset = load_dataset(args.split, args.dataset_name, args.path, batch_size=args.batch_size, extra_noise=args.extra_noise, noise_type=args.noise_type, noise_level=args.noise_level)

    print("========================== Start! ==========================")
    print("Strategy name: ", config["tta_name"])
    print("Dataset name: ", config["dataset_name"])

    gt_texts = []
    transcriptions = []
    memory_usage = []
    wers = []
    count = 0
    for sample in tqdm(dataset):
        count += 1
        lens, wavs, texts, files = sample
        torch.cuda.reset_peak_memory_stats()
        trans = tta.run(wavs)
        gt_texts.append(texts[0])
        transcriptions.append(trans)
        print("original text: ", texts[0])
        print("transcription: ", trans)
        print(f'memory: {torch.cuda.max_memory_allocated()/(1024*1024):.3f}MB')
        memory_usage.append(torch.cuda.max_memory_allocated()/(1024*1024))
        
        torch.cuda.empty_cache()

    wer_value = jiwer.wer(gt_texts, transcriptions)
    print("WER: ", wer_value)

    results = {
        'asr': config["system_config"]["model_name"],
        'dataset_name': config["dataset_name"],
        'dataset_num': len(dataset),
        'split': config["split"],
        'memory_usage': memory_usage,
        'final_wer': wer_value,
        'avg_memory_usage': np.mean(memory_usage),
        'max_memory_usage': np.max(memory_usage),
        'min_memory_usage': np.min(memory_usage),
    }

    import json

    output_file = f'/home/jiahengd/tta-suta/Accuracy-efficency-results/{config["tta_name"].upper()}/{results["asr"]}_{results["dataset_name"]}_{results["split"]}'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('--tta_name', type=str, default="suta")
    parser.add_argument('--dataset_name', type=str, default="chime")
    parser.add_argument('--config', type=str, default="config/system/suta-based.yaml")
    parser.add_argument('--tta_config', nargs='+', default=["config/tta/suta.yaml"])
    parser.add_argument('--split', type=str, default=["test-other"])
    parser.add_argument('--path', type=str, default="/home/jiahengd/tta-suta/LibriSpeech")
    parser.add_argument('--noise_type', type=str, default="")
    parser.add_argument('--noise_level', type=int, default=0)
    parser.add_argument('--extra_noise', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()
    print(args.tta_name)
    
    seed_everything(42)
    main(args)
