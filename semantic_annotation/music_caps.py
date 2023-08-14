import os
import ast
import json
import argparse
import transformers
import pandas as pd
from datasets import Dataset, DatasetDict

from utils.tag_processor import music_caps_tag_processor, song_describer_tag_processor
from utils.multitag_processor import music_caps_multitag_processor, song_describer_multitag_processor
from utils.caption_processor import music_caps_caption_processor, song_describer_caption_processor

def get_query2target_idx(query2target, target2idx):
    query2target_idx = {}
    for query, target_list in query2target.items():
        query2target_idx[query] = [target2idx[i] for i in target_list]
    return query2target_idx

def get_ground_truth(query2target, query2idx, query2target_idx, target2idx):
    query2target_gt = []
    for query, target_id in query2target.items():
        query_idx = query2idx[query]
        target_idx = query2target_idx[query]
        query2target_gt.append(
            {
                "query": query,
                "query_idx": query_idx,
                "target_id": target_id,
                "target_idx": target_idx
            }
        )
    return query2target_gt

def retrieval_target_generation(annotations, tokenizer, threshold=10):
    tag2track_matrix, track2tag, tag2track, tag_map = music_caps_tag_processor(annotations, tokenizer, threshold)
    _, track2multitag, multitag2track = music_caps_multitag_processor(annotations, tokenizer)
    _, track2caption, caption2track = music_caps_caption_processor(annotations)
    # unique_entity 
    unique_track = list(track2caption.keys())
    unique_caption =  list(caption2track.keys())
    unique_multitag = list(multitag2track.keys())
    unique_tag = list(tag2track.keys())
    # entity2idx
    track2idx = {i:idx for idx, i in enumerate(unique_track)}
    caption2idx = {i:idx for idx, i in enumerate(unique_caption)}
    multitag2idx = {i:idx for idx, i in enumerate(unique_multitag)}
    tag2idx = {i:idx for idx, i in enumerate(unique_tag)}
    
    caption2track_idx = get_query2target_idx(caption2track, track2idx)
    track2caption_idx = get_query2target_idx(track2caption, caption2idx)

    multitag2track_idx = get_query2target_idx(multitag2track, track2idx)
    track2multitag_idx = get_query2target_idx(track2multitag, multitag2idx)

    tag2track_idx = get_query2target_idx(tag2track, track2idx)
    track2tag_idx = get_query2target_idx(track2tag, tag2idx)

    caption2music_gt = get_ground_truth(caption2track, caption2idx, caption2track_idx, track2idx)
    track2caption_gt = get_ground_truth(track2caption, track2idx, track2caption_idx, caption2idx)

    multitag2music_gt = get_ground_truth(multitag2track, multitag2idx, multitag2track_idx, track2idx)
    track2multitag_gt = get_ground_truth(track2multitag, track2idx, track2multitag_idx, multitag2idx)

    tag2music_gt = get_ground_truth(tag2track, tag2idx, tag2track_idx, track2idx)
    track2tag_gt = get_ground_truth(track2tag, track2idx, track2tag_idx, tag2idx)

    music_caps_retrieval = {
        "caption2music": Dataset.from_list(caption2music_gt),
        "music2caption": Dataset.from_list(track2caption_gt),
        "multitag2music": Dataset.from_list(multitag2music_gt),
        "music2multitag": Dataset.from_list(track2multitag_gt),
        "tag2music": Dataset.from_list(tag2music_gt),
        "music2tag": Dataset.from_list(track2tag_gt),
    }
    mc_dataset = DatasetDict(music_caps_retrieval)

    # if you wanna save dataset
    for split, dataset in dataset.items():
        dataset.to_json(f"music_caps_retrieval_{split}.jsonl")


def load_metadata():
    annotations = pd.read_csv("../datasets/music_caps/musiccaps-public.csv")
    annotations = annotations[annotations["is_audioset_eval"]] # parsing evaluation
    annotations = annotations[annotations["ytid"] != "VRxFYwbik6A"] # delete black list
    annotations["aspect_list"] = annotations["aspect_list"].apply(ast.literal_eval)
    return annotations

if __name__ == "__main__":
    annotations = load_metadata()
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    retrieval_target_generation(annotations, tokenizer, threshold=10)

    
    

    
