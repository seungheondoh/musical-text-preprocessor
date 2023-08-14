import os
import glob
import yaml
import tqdm
import random
import librosa
import argparse
import numpy as np
import soundfile as sf
from collections import defaultdict
import json
import pandas as pd
import multiprocessing
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

class Generator:
    def __init__(self, data_path="./dataset/slakh2100_flac_redux", split="test"):
        super(Generator, self).__init__()
        random.seed(42)
        self.data_path = data_path
        self.split = split
        self.twelve_instruments = [
            "Bass",
            "Brass",
            "Chromatic Percussion",
            "Drums",
            "Guitar",
            "Organ",
            "Piano",
            "Pipe",
            "Reed",
            "Strings",
            "Synth Lead",
            "Synth Pad"
        ]
        # get slakh data list
        self.interval = 1
        if self.split:
            self.metadata_files = glob.glob((os.path.join(data_path, self.split, "*/metadata.yaml")))
        else:
            self.metadata_files = glob.glob((os.path.join(data_path, "*/*/metadata.yaml")))
        
        self.track_names = list(set((fn.split("/")[-2] for fn in self.metadata_files)))
        
    def get_stems(self, metadata_fn):
        # read yaml
        with open(metadata_fn, "r") as f:
            metadata = yaml.safe_load(f)

        # read audio
        stems = defaultdict(list)
        for inst_key in metadata["stems"].keys():
            inst_cls = metadata["stems"][inst_key]["inst_class"]
            if inst_cls in self.twelve_instruments:
                stem_path = os.path.join(
                    os.path.dirname(metadata_fn),
                    "stems",
                    inst_key + ".flac"
                )
                if os.path.exists(stem_path):
                    x, sr = librosa.load(stem_path)
                    stems[inst_cls].append(x)
        for inst in stems.keys():
            stems[inst] = np.sum(stems[inst], axis=0)
        return stems, list(stems.keys()), sr

    def generate_examples(self, index):
        metadata_fn = self.metadata_files[index]
        track_name = metadata_fn.split("/")[-2]
        track_path = os.path.join(self.data_path, "slakh_energy")
        stems, instruments, original_fs = self.get_stems(metadata_fn)
        inst_energy = {}
        for i, inst in enumerate(instruments):
            x = stems[inst]
            duration = int(len(x)/original_fs)
            interval_sample = int(self.interval * original_fs)
            energy = {}
            for idx in range(0, duration - self.interval):
                start = int(idx * original_fs)
                end = start + interval_sample
                energy[idx] = int(np.abs(x[ start : end ]).sum())
            inst_energy[inst] = energy
        return {"track_name": track_name, "inst_energy": inst_energy}
        
    def get_best_engergy_bin(self, energy_dict):
        window = 30
        df = pd.DataFrame(energy_dict)
        moving_averages = df.rolling(window).mean().shift(-window + 1).fillna(0)
        target_idx = moving_averages.mean(axis=1).argmax()
        return target_idx
        
    def generate(self):
        with poolcontext(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self.generate_examples, range(len(self.metadata_files)))
        slakh_idx = {i["track_name"]: str(self.get_best_engergy_bin(i['inst_energy'])) for i in results}
        with open("slakh_idx.json", mode="w") as io:
            json.dump(slakh_idx, io, indent=4)
        with open("slakh_energy.jsonl",mode="w") as file: 
        	for i in results: file.write(json.dumps(i) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to Slakh data")
    args = parser.parse_args()
    generator = Generator()
    generator.generate()

