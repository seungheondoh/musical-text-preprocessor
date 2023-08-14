import os
import glob
import yaml
import tqdm
import random
import openai
import librosa
import argparse
import numpy as np
import soundfile as sf
from collections import defaultdict
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
    def __init__(self, args, split="test"):
        super(Generator, self).__init__()
        random.seed(42)
        self.data_path = os.path.join(args.data_path, "slakh2100_flac_redux")
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
        self.window = int(args.window)
        openai.api_key = args.apikey

        self.metadata_files = glob.glob((os.path.join(data_path, self.split, "*/metadata.yaml")))
        self.track_names = list(set((fn.split("/")[-2] for fn in self.metadata_files)))
        self.fname2idx = json.load(open(os.path.join(self.data_path, "assets/slakh_idx.json"), 'r'))

        
    def get_stems(self, metadata_fn, start_idx):
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
                    x, original_fs = librosa.load(stem_path)
                    start = int(start_idx * original_fs)
                    interval = int(self.window * original_fs)
                    end = start + interval

                    if abs(x[start : end]).sum() > 100:  # if stem is not empty
                        stems[inst_cls].append(x[start : end])
        # sum audio
        if len(stems.keys()) < 2: # less than two valid instruments
            return False, 0, 0, 0

        for inst in stems.keys():
            stems[inst] = np.sum(stems[inst], axis=0)
        return True, stems, list(stems.keys()), original_fs

    @staticmethod
    def get_captions(inst_a, inst_b, order):
        prefix_prompt = "I want to write music captions that describe the temporal order of two instruments. Write this caption in three styles"
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
    
    def mix_audio(self, stem_a, stem_b, original_fs):
        interval = int(self.window * original_fs)
        mix_point = int(int(self.window / 2) * original_fs)
        mask = np.ones(interval)
        mask[:mix_point] = 0
        return stem_a + (stem_b * mask)

    def generate_examples(self, index):
        metadata_fn = self.metadata_files[index]
        track_name = metadata_fn.split("/")[-2]
        start_idx = int(self.fname2idx[track_name])
        track_path = os.path.join(self.data_path, "slakh_temporal", track_name)

        if os.path.exists(track_path):
            print("%s already exists" % track_name)
            is_valid = None
        else:
            is_valid, stems, instruments, original_fs = self.get_stems(metadata_fn, start_idx)

        if is_valid:
            inst_pairs = list(combinations(instruments, 2))
            for pair in inst_pairs:
                inst_a, inst_b = pair
                captions_ab = self.get_captions(inst_a, inst_b, "before")
                captions_ba = self.get_captions(inst_a, inst_b, "after")

                mixed_wav_ab = self.mix_audio(stems[inst_a], stems[inst_b], original_fs)
                mixed_wav_ba = self.mix_audio(stems[inst_b], stems[inst_a], original_fs)

                # save audio
                os.makedirs(
                    track_path,
                    exist_ok=True
                )
                wav_ab_fn = os.path.join(
                    self.data_path, "slakh_temporal", track_name,
                    "%s_%s.wav" % (inst_a, inst_b)
                )
                wav_ba_fn = os.path.join(
                    self.data_path, "slakh_temporal", track_name,
                    "%s_%s.wav" % (inst_b, inst_a)
                )
                sf.write(wav_ab_fn, mixed_wav_ab, original_fs)
                sf.write(wav_ba_fn, mixed_wav_ba, original_fs)

                # save captions
                caption_ab_fn = os.path.join(
                    self.data_path, "slakh_temporal", track_name, 
                    "%s_%s.json" % (inst_a, inst_b)
                )
                caption_ba_fn = os.path.join(
                    self.data_path, "slakh_temporal", track_name, 
                    "%s_%s.json" % (inst_b, inst_a)
                )
                with open(caption_ab_fn, mode="w") as io:
                    json.dump(captions_ab, io, indent=4)
                with open(caption_ba_fn, mode="w") as io:
                    json.dump(captions_ba, io, indent=4)

    def generate(self):
        with poolcontext(processes=multiprocessing.cpu_count()) as pool:
            pool.map(self.generate_examples, range(len(self.metadata_files)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to Slakh data", type=str, default="../dataset")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--apikey", type=str, default="your api key")
    args = parser.parse_args()
    generator = Generator(args)
    generator.generate()