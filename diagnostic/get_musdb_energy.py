import os
import musdb
import argparse
import numpy as np
import soundfile as sf
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
    def __init__(self, args):
        super(Generator, self).__init__()
        self.data_path = "./dataset"
        self.interval = 1
        self.db = musdb.DB(os.path.join(self.data_path, "musdb18"), subsets="test", is_wav=True)
        print(len(self.db))

    def generate_examples(self, index):
        fname = str(self.db.tracks[index])
        track_name = self.db.tracks[index].title
        instruments = list(self.db.tracks[index].sources.keys())
        track_path = os.path.join(self.data_path, "musdb_energy")
        inst_energy = {}
        for i, inst in enumerate(instruments):
            x = self.db.tracks[index].targets[inst].audio.mean(axis=-1)
            original_fs = self.db.tracks[index].targets[inst].rate
            duration = int(len(x)/original_fs)
            interval_sample = int(self.interval * original_fs)
            
            energy = {}
            for idx in range(0, duration - self.interval):
                start = int(idx * original_fs)
                end = start + interval_sample
                energy[idx] = int(np.abs(x[ start : end ]).sum())
            inst_energy[inst] = energy
        return {"track_name": fname, "inst_energy": inst_energy}
    
    def get_best_engergy_bin(self, energy_dict):
        window = 30
        df = pd.DataFrame(energy_dict)
        moving_averages = df.rolling(window).mean().shift(-window + 1).fillna(0)
        target_idx = moving_averages.mean(axis=1).argmax()
        return target_idx


    def generate(self):
        with poolcontext(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self.generate_examples, range(50))
        musdb_idx = {i["track_name"]: str(self.get_best_engergy_bin(i['inst_energy'])) for i in results}
        with open("musdb_idx.json", mode="w") as io:
            json.dump(musdb_idx, io, indent=4)
        with open("musdb_energy.jsonl",mode="w") as file: 
        	for i in results: file.write(json.dumps(i) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to MUSDB data")
    
    args = parser.parse_args()
    generator = Generator(args)
    generator.generate()
