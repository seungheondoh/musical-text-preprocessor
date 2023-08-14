import os
import json
import tqdm
import random
import musdb
import argparse
import numpy as np
import soundfile as sf
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
        random.seed(42)
        self.data_path = args.data_path
        self.window = int(args.window)
        openai.api_key = args.api_key
        # get musdb data
        self.db = musdb.DB(os.path.join(self.data_path, "musdb18"), subsets="test", is_wav=True)
        self.fname2idx = json.load(open(os.path.join(self.data_path, "assets/musdb_idx.json"), 'r'))

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

    def mix_audio(self, index, pos, start_idx):
        wavs = []
        is_zero = False
        for inst in pos:
            # to mono
            x = self.db.tracks[index].targets[inst].audio.mean(axis=-1)
            original_fs = self.db.tracks[index].targets[inst].rate
            start = int(start_idx * original_fs)
            interval = int(self.window * original_fs)
            end = start + interval
            # check if the instrument channel is non-zero (first 30s)
            if np.abs(x[start : end]).sum() < 100:
                is_zero = True
            wavs.append(x)
        return np.array(wavs).sum(0)[start : end], original_fs, is_zero
    
    def generate_examples(self, index):
        fname = str(self.db.tracks[index])
        start_idx = int(self.fname2idx[fname])
        instruments = list(self.db.tracks[index].sources.keys())
        instruments.remove("other")
        track_path = os.path.join(self.data_path, "musdb_negation", fname)

        if os.path.exists(track_path):
            print("%s already exists" % fname)
        else:
            if len(instruments) > 1:
                # randomly remove one item
                for i, inst in enumerate(instruments):
                    neg = [inst]
                    pos = instruments[:i] + instruments[i+1:]
                    caption_dict = self.get_captions(pos, neg)
                    mixed_wav, original_fs, is_zero = self.mix_audio(index, pos, start_idx)
                    caption_dict["start_idx"] = start_idx
                    if not is_zero:
                        os.makedirs(
                            os.path.join(self.data_path, "musdb_negation", fname),
                            exist_ok=True
                        )
                        wav_fn = os.path.join(
                            self.data_path, "musdb_negation", fname,
                            "no_%s.wav" % inst
                        )
                        sf.write(wav_fn, mixed_wav, original_fs)

                        caption_fn = os.path.join(
                            self.data_path, "musdb_negation", fname,
                            "no_%s.json" % inst
                        )
                        with open(caption_fn, mode="w") as io:
                            json.dump(caption_dict, io, indent=4)
                        

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
