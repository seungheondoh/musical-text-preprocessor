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
import openai
import json
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
    def get_captions(pos, neg):
        random.shuffle(pos)
        neg_text = neg[0]
        pos_text = ", ".join(pos)
        prompt = "Re-write this music caption in three different styles. You can mix the order of the instruments."
        inputs = "This song includes %s, but without %s.\n" % (pos_text, neg_text)
        query = f"{prompt}: {inputs}"
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
            "neg": neg,
            "pos": pos,
            "prompt": prompt,
            "inputs": inputs,
            "captions": captions
            }


    def generate_examples(self, index):
        metadata_fn = self.metadata_files[index]
        track_name = metadata_fn.split("/")[-2]
        start_idx = int(self.fname2idx[track_name])
        track_path = os.path.join(self.data_path, "slakh_negation", track_name)

        if os.path.exists(track_path):
            print("%s already exists" % track_name)
            is_valid = None
        else:
            is_valid, stems, instruments, original_fs = self.get_stems(metadata_fn, start_idx)

        if is_valid:
            is_valid, stems, instruments, original_fs = self.get_stems(metadata_fn, start_idx)
            # randomly remove one item
            for i, inst in enumerate(instruments):
                neg = [inst]
                pos = instruments[:i] + instruments[i + 1:]
                caption_dict = self.get_captions(pos, neg)
                mixed_wav = np.sum([stems[inst] for inst in pos], axis=0)
                caption_dict["start_idx"] = start_idx

                # save audio
                os.makedirs(
                    os.path.join(self.data_path, "slakh_negation", track_name), 
                    exist_ok=True
                )
                wav_fn = os.path.join(
                    self.data_path, "slakh_negation", track_name,
                    "no_%s.wav" % inst
                )
                sf.write(wav_fn, mixed_wav, original_fs)

                # save captions
                caption_fn = os.path.join(
                    self.data_path, "slakh_negation", track_name, 
                    "no_%s.json" % inst
                )
                with open(caption_fn, mode="w") as io:
                    json.dump(caption_dict, io, indent=4)

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

