import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict

text_per_audio = 3
def _bulid_negation():
    root_dir = "../datasets/musdb_negation"
    musdb_negation = []
    for fname in os.listdir(root_dir):
        audios = [i.replace(".wav", "") for i in os.listdir(f"{root_dir}/{fname}") if ".wav" in i]
        pos_inst, captions = [], []
        for audio in audios:
            annotation = json.load(open(f"{root_dir}/{fname}/{audio}.json",'r'))
            start_sec = annotation['start_idx']
            pos_inst.append(annotation["pos"])
            caption = [text.replace("1.","").replace("2.","").replace("3.","").strip() for text in annotation['captions'].strip().split("\n") if len(text) > 10] # for drop `Style 1:`
            captions.append(caption)
        musdb_negation.append({
            "track_id": fname,
            "num_audio": len(audios),
            "num_text": len(audios) * text_per_audio,
            "text_per_audio": text_per_audio,
            "audios": audios,
            "texts": captions,
            "inst_label": pos_inst,
            "start_s": start_sec,
            "duration": 30
        })
    return musdb_negation

def _bulid_temporal_ordering():
    root_dir = "../datasets/musdb_temporal"
    musdb_temporal = []
    for fname in os.listdir(root_dir):
        audios = [i.replace(".wav", "") for i in os.listdir(f"{root_dir}/{fname}") if ".wav" in i]
        pos_inst, captions = [], []
        for audio in audios:
            annotation = json.load(open(f"{root_dir}/{fname}/{audio}.json",'r'))
            start_sec = annotation['start_idx']
            pos_inst.append([annotation["inst_a"], annotation["inst_b"]])
            caption = [text.replace("1.","").replace("2.","").replace("3.","").strip() for text in annotation['captions'].split("\n") if len(text) > 0]
            captions.append(caption)
        musdb_temporal.append({
            "track_id": fname,
            "num_audio": len(audios),
            "num_text": len(audios) * text_per_audio,
            "text_per_audio": text_per_audio,
            "audios": audios,
            "texts": captions,
            "inst_label": pos_inst,
            "start_s": start_sec,
            "duration": 30
        })
    return musdb_temporal
        

def bulid_annotation():
    musdb_negation = _bulid_negation()
    musdb_temporal = _bulid_temporal_ordering()
    musdb_dataset = {
        "negation": Dataset.from_list(musdb_negation),
        "temporal_ordering": Dataset.from_list(musdb_temporal)
    }
    musdb_datadict = DatasetDict(musdb_dataset)
    # musdb_datadict.push_to_hub("mulab/diagnostic_eval_musdb")

if __name__ == "__main__":
    bulid_annotation()

