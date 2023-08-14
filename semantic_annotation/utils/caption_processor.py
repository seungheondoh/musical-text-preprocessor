import os
import re
import ast
import transformers
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

flatten_list = lambda lst: [item for sublist in lst for item in sublist]

def group_to_list(group):
    return list(group)

def _get_ground_turth(annotations, idx_col, caption_col):
    df_grouped = annotations.groupby(idx_col)[caption_col].apply(group_to_list)
    mlb = MultiLabelBinarizer()
    binarys = mlb.fit_transform(list(df_grouped))
    df_binary = pd.DataFrame(
        binarys, index=list(df_grouped.index), columns=mlb.classes_
    )
    track_to_caption, caption_to_track = {}, {}
    for caption in df_binary:
        track_list = list(df_binary[df_binary[caption] == 1].index)
        caption_to_track[caption] = list(set(track_list))

    for idx in range(len(df_binary)):
        instance = df_binary.iloc[idx]
        track_to_caption[instance.name] = list(instance[instance==1].index)
    return df_binary, track_to_caption, caption_to_track

def song_describer_caption_processor(annotations):
    """
    song_describer, get tag_based_rerieval groundturth
    """
    df_binary, track_to_caption, caption_to_track = _get_ground_turth(
        annotations = annotations, 
        idx_col = "track_id", 
        caption_col = "caption"
        )
    return df_binary, track_to_caption, caption_to_track


def music_caps_caption_processor(annotations):
    """
    music_caps, get tag_based_rerieval groundturth
    """
    annotations = annotations[annotations["is_audioset_eval"]]
    df_binary, track_to_caption, caption_to_track = _get_ground_turth(
        annotations = annotations, 
        idx_col = "ytid", 
        caption_col = "caption"
    )
    return df_binary, track_to_caption, caption_to_track
