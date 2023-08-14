import os
import musdb
import argparse
import numpy as np
import soundfile as sf
from itertools import combinations

import json
import openai
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
        self.data_path = args.data_path
        self.window = int(args.window)
        openai.api_key = args.apikey
        # get musdb data
        self.db = musdb.DB(os.path.join(self.data_path, "musdb18"), subsets="test", is_wav=True)
                self.fname2idx = json.load(open(os.path.join(self.data_path, "assets/musdb_idx.json"), 'r'))

    @staticmethod
    def get_captions(inst_a, inst_b, order):
        prefix_prompt = "I want to write music captions that describe the temporal order of two instruments. Write this caption in three styles",
        sufix_prompt = "The answers need to be three sentences parsed with newlines. Do not number them. Do not add subjective information. Do not add any additional information than the temporal order.\n"
        inputs = "%s is played %s %s." % (inst_a, order, inst_b)
        query = f"{prefix_prompt}: {inputs} {sufix_prompt}"
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=query,
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        captions = response["choices"][0]["text"]
        return {
            "inst_a": inst_a,
            "inst_b": inst_b,
            "order": order,
            "prefix_prompt": prefix_prompt,
            "inputs": inputs,
            "sufix_prompt": sufix_prompt,
            "captions": captions
            }

    def mix_audio(self, index, inst_a, inst_b, start_idx):
        is_zero = False
        # to mono
        x_a = self.db.tracks[index].targets[inst_a].audio.mean(axis=-1)
        x_b = self.db.tracks[index].targets[inst_b].audio.mean(axis=-1)
        original_fs = self.db.tracks[index].targets[inst_a].rate
        interval = int(self.window * original_fs)
        mix_point = int(int(self.window / 2) * original_fs)

        start = int(start_idx * original_fs)
        mid = start + mix_point
        end = start + interval
        # check if the instrument channel is non-zero (first 15s of x_a and 15-30s of x_b)
        if (np.abs(x_a[start : mid]).sum() < 100) or (np.abs(x_b[mid :end]).sum() < 100):
            is_zero = True
        # mute first 15 sec of x_b
        x_b[start:mid] = 0.0
        return (x_a + x_b)[start:end], is_zero, original_fs
    
    def generate_examples(self, index):
        fname = str(self.db.tracks[index])
        start_idx = int(self.fname2idx[fname])
        track_name = self.db.tracks[index].title
        instruments = list(self.db.tracks[index].sources.keys())
        instruments.remove("other")
        track_path = os.path.join(self.data_path, "musdb_temporal", track_name)
        if os.path.exists(track_path):
            print("%s already exists" % track_name)
        else:
            inst_pairs = list(combinations(instruments, 2))
            for pair in inst_pairs:
                inst_a, inst_b = pair    
                captions_ab = self.get_captions(inst_a, inst_b, "before")
                captions_ba = self.get_captions(inst_a, inst_b, "after")
                captions_ab["start_idx"] = start_idx
                captions_ba["start_idx"] = start_idx

                mixed_wav_ab, is_zero, original_fs = self.mix_audio(index, inst_a, inst_b, start_idx)
                mixed_wav_ba, is_zero, original_fs = self.mix_audio(index, inst_b, inst_a, start_idx)
                
                if not is_zero:
                    os.makedirs(os.path.join(self.data_path, "musdb_temporal", fname), exist_ok=True)
                    wav_ab_fn = os.path.join(self.data_path, "musdb_temporal", fname, "%s_%s.wav" % (inst_a, inst_b))
                    wav_ba_fn = os.path.join(self.data_path, "musdb_temporal", fname, "%s_%s.wav" % (inst_b, inst_a))
                    sf.write(wav_ab_fn, mixed_wav_ab, original_fs)
                    sf.write(wav_ba_fn, mixed_wav_ba, original_fs)

                    caption_ab_fn = os.path.join(self.data_path, "musdb_temporal", fname, "%s_%s.json" % (inst_a, inst_b))
                    caption_ba_fn = os.path.join(self.data_path, "musdb_temporal", fname, "%s_%s.json" % (inst_b, inst_a))
                    with open(caption_ab_fn, mode="w") as io:
                        json.dump(captions_ab, io, indent=4)
                    with open(caption_ba_fn, mode="w") as io:
                        json.dump(captions_ba, io, indent=4)

    def generate(self):
        with poolcontext(processes=multiprocessing.cpu_count()) as pool:
            pool.map(self.generate_examples, range(50))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to MUSDB data", type=str, default="../dataset")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--apikey", type=str, default="your api key")
    args = parser.parse_args()
    generator = Generator(args)
    generator.generate()
