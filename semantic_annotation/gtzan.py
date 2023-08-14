import os
import ast
import json
import argparse
import transformers
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelBinarizer
from datasets import Dataset, DatasetDict
from utils.io_utils import save_idx

def get_retrieval_taget(annotations, index_col, label_col):
    lb = LabelBinarizer()
    annotations = [i for i in annotations if len(i[label_col]) > 0]
    binarys = lb.fit_transform([i[label_col] for i in annotations])
    df_binary = pd.DataFrame(
        binarys, index=[i[index_col] for i in annotations], columns=lb.classes_
    )
    track_to_label, label_to_track = {}, {}
    for label in df_binary:
        track_list = list(df_binary[df_binary[label] == 1].index)
        label_to_track[label] = list(set(track_list))

    for idx in range(len(df_binary)):
        instance = df_binary.iloc[idx]
        track_to_label[instance.name] = list(instance[instance == 1].index)
    return track_to_label, label_to_track

def get_query2target_idx(query2target, target2idx):
    query2target_idx = {}
    for query, target_list in query2target.items():
        query2target_idx[query] = [target2idx[i] for i in target_list]
    return query2target_idx

def load_metadata():
    gtzan_bind = load_dataset("seungheondoh/gtzan-bind")
    return list(gtzan_bind["gtzan_bind_v1"])


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

def retrieval_target_generation(annotations):
    unique_track = list(set([i["track_id"] for i in annotations]))
    unique_artist = list(set([i["artist_name"] for i in annotations]))
    unique_album = list(set([i["album"] for i in annotations]))
    unique_caption = list(set([i["caption_15s"] for i in annotations]))
    unique_tag = list(set([i["tag"] for i in annotations]))
    unique_key = list(set([i["key"] for i in annotations]))
    quantize_tempo = [str(round(float(i["tempo_mean"]), -1)) for i in annotations]
    for q_tempo, item in zip(quantize_tempo, annotations):
        item["quantize_tempo"] = q_tempo
    unique_tempo = list(set(quantize_tempo))

    track2idx = {i:idx for idx, i in enumerate(unique_track)}
    artist2idx = {i:idx for idx, i in enumerate(unique_artist)}
    album2idx = {i:idx for idx, i in enumerate(unique_album)}
    caption2idx = {i:idx for idx, i in enumerate(unique_caption)}
    tag2idx = {i:idx for idx, i in enumerate(unique_tag)}
    key2idx = {i:idx for idx, i in enumerate(unique_key)}
    tempo2idx = {i:idx for idx, i in enumerate(unique_tempo)}
    
    list_of_dict = [track2idx ,artist2idx ,album2idx ,caption2idx ,tag2idx ,key2idx ,tempo2idx]
    list_of_dict_name = ["track2idx" ,"artist2idx" ,"album2idx" ,"caption2idx" ,"tag2idx" ,"key2idx" ,"tempo2idx"]
    save_idx(list_of_dict, list_of_dict_name)
    
    track2artist, artist2track = get_retrieval_taget(annotations, index_col="track_id", label_col="artist_name")
    track2album, album2track = get_retrieval_taget(annotations, index_col="track_id", label_col="album")
    track2caption, caption2track = get_retrieval_taget(annotations, index_col="track_id", label_col="caption_15s")
    track2tag, tag2track = get_retrieval_taget(annotations, index_col="track_id", label_col="tag")
    track2key, key2track = get_retrieval_taget(annotations, index_col="track_id", label_col="key")
    track2tempo, tempo2track = get_retrieval_taget(annotations, index_col="track_id", label_col="quantize_tempo")
    
    artist2track_idx = get_query2target_idx(artist2track, track2idx)
    album2track_idx = get_query2target_idx(album2track, track2idx)
    caption2track_idx = get_query2target_idx(caption2track, track2idx)
    tag2track_idx = get_query2target_idx(tag2track, track2idx)
    key2track_idx = get_query2target_idx(key2track, track2idx)
    tempo2track_idx = get_query2target_idx(tempo2track, track2idx)

    artist2music_gt = get_ground_truth(artist2track, artist2idx, artist2track_idx, track2idx)
    album2music_gt = get_ground_truth(album2track, album2idx, album2track_idx, track2idx)
    caption2music_gt = get_ground_truth(caption2track, caption2idx, caption2track_idx, track2idx)
    tag2music_gt = get_ground_truth(tag2track, tag2idx, tag2track_idx, track2idx)
    key2music_gt = get_ground_truth(key2track, key2idx, key2track_idx, track2idx)
    tempo2music_gt = get_ground_truth(tempo2track, tempo2idx, tempo2track_idx, track2idx)

    gtzan_retrieval = {
        "artist2music": Dataset.from_list(artist2music_gt),
        "album2music": Dataset.from_list(album2music_gt),
        "caption2music": Dataset.from_list(caption2music_gt),
        "tag2music": Dataset.from_list(tag2music_gt),
        "key2music": Dataset.from_list(key2music_gt),
        "tempo2music": Dataset.from_list(tempo2music_gt),
    }
    # gtzan_dataset = DatasetDict(gtzan_retrieval)
    # gtzan_dataset.push_to_hub("seungheondoh/gtzan_retrieval")

if __name__ == "__main__":
    # args = argparse()
    annotations = load_metadata()
    retrieval_target_generation(annotations)
    