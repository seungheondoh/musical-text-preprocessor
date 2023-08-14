import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict

text_per_audio = 3
def _bulid_negation():
    root_dir = "../datasets/slakh_negation"
    slakh_negation = []
    for fname in os.listdir(root_dir):
        audios = [i.replace(".wav", "") for i in os.listdir(f"{root_dir}/{fname}") if ".wav" in i]
        pos_inst, captions = [], []
        for audio in audios:
            annotation = json.load(open(f"{root_dir}/{fname}/{audio}.json",'r'))
            start_sec = annotation['start_idx']
            pos_inst.append(annotation["pos"])
            caption = [text.replace("1.","").replace("2.","").replace("3.","").strip() for text in annotation['captions'].strip().split("\n") if len(text) > 10] # for drop `Style 1:`
            if len(caption) != text_per_audio:
                raise print("error!")
                
            captions.append(caption)
        slakh_negation.append({
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
    return slakh_negation

def merge_lines(style_text):
    cap1 = style_text.split("Style 1:")[1].split("Style 2:")[0].replace("\n", "")
    cap2 = style_text.split("Style 2:")[1].split("Style 3:")[0].replace("\n", "")
    cap3 = style_text.split("Style 3:")[1].replace("\n", "")
    return [cap1, cap2, cap3]

def _bulid_temporal_ordering():
    root_dir = "../datasets/slakh_temporal"
    slakh_temporal = []
    for fname in os.listdir(root_dir):
        audios = [i.replace(".wav", "") for i in os.listdir(f"{root_dir}/{fname}") if ".wav" in i]
        pos_inst, captions = [], []
        for audio in audios:
            annotation = json.load(open(f"{root_dir}/{fname}/{audio}.json",'r'))
            start_sec = int(annotation['start_idx'])
            pos_inst.append([annotation["inst_a"], annotation["inst_b"]])
            caption = [text.replace("1.","").replace("2.","").replace("3.","").strip() for text in annotation['captions'].split("\n") if len(text) > 10]
            if len(caption) > 3:
                save_caption = [cap for cap in caption if (annotation["inst_a"] in cap) and (annotation["inst_b"] in cap)]
                sorted_list = sorted(save_caption, key=len)
                caption = sorted_list[:3]
                if len(caption) < 3:
                    caption = merge_lines(annotation['captions'])
            if len(caption) != text_per_audio:
                raise print("error!")
            captions.append(caption)
        slakh_temporal.append({
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
    return slakh_temporal
        

def bulid_annotation():
    slakh_negation = _bulid_negation()
    slakh_temporal = _bulid_temporal_ordering()
    slakh_dataset = {
        "negation": Dataset.from_list(slakh_negation),
        "temporal_ordering": Dataset.from_list(slakh_temporal)
    }
    slakh_datadict = DatasetDict(slakh_dataset)
    slakh_datadict.push_to_hub("mulab/diagnostic_eval_slakh")

if __name__ == "__main__":
    bulid_annotation()

