import re
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

flatten_list = lambda lst: [item for sublist in lst for item in sublist]


def normalize_text(text):
    """
    Text normalization for alphabets and numbers
        args: text: list of tag
    """
    pattern = r"[^a-zA-Z0-9\s]"
    removed_text = re.sub(pattern, "", text)
    return removed_text


def _generate_tag_map(tag_list, tokens):
    """
    Generate tag merge dict
        args:
            tag_list: list of tag
            tokens : subword token (BPE, SentencePiece)
        return:
            tag_map: Dict (original tag: merge tag)
    """
    tag_map = {}
    for tag_q, token_q in zip(tqdm(tag_list), tokens):
        for tag_p, token_p in zip(tag_list, tokens):
            if (tag_q == tag_p) or (tag_p in tag_map):
                pass
            else:
                if set(token_q) == set(token_p):
                    if len(tag_q) > len(tag_p):
                        tag_map[tag_q] = tag_p
                    else:
                        tag_map[tag_p] = tag_q
    return tag_map


def _apply_tag_map(aspect_list, tag_map):
    """
    Delete synonym with tag map
        args:
            aspect_list: list of tag
            tag_map : tag merge dictionary
        return:
            norm_lists: list of normalize tag
    """
    norm_lists = []
    for tag_list in aspect_list:
        norm_list = []
        for tag in tag_list:
            tag = normalize_text(tag)
            if tag in tag_map:
                norm_list.append(tag_map[tag])
            else:
                norm_list.append(tag)
        norm_lists.append(norm_list)
    return norm_lists


def tag_thresholding(df_binary, threshold):
    # drop by tag
    df_sum = df_binary.sum(axis=0)
    df_sum = df_sum[df_sum >= threshold]
    df_binary = df_binary[df_sum.index]
    
    # drop by columns
    df_sum = df_binary.sum(axis=1)
    df_sum = df_sum[df_sum > 0] # more than one annotation
    df_binary = df_binary.loc[df_sum.index]

    min_tag = df_binary.sum(axis=0).min()
    min_track = df_binary.sum(axis=1).min()
    return df_binary, min_tag, min_track

def iterable_drop(df_binary, threshold):
    while True:
        df_binary, min_tag, min_track = tag_thresholding(df_binary, threshold)
        if (min_tag >= threshold) and (min_track > 0):
            print("converge iterable process, ", "\n min tag:",min_tag, "\n min track:", min_track, "\n annotation shape", df_binary.shape)
            break
    return df_binary

    
def _get_retrieval_ground_turth(aspect_list, annotations, idx_col, threshold):
    """
    generate tag to track dictionary
    args:
        aspect_list: list of tag
        annotations: metadata pandas dataframe
        idx_col: column of unique track identifier (ytid, track_id)
        threshold :
    return:
        norm_lists: list of normalize tag
    """
    mlb = MultiLabelBinarizer()
    binarys = mlb.fit_transform(aspect_list)
    df_binary = pd.DataFrame(
        binarys, index=list(annotations[idx_col]), columns=mlb.classes_
    )
    df_binary = df_binary[~df_binary.index.duplicated(keep='first')]
    df_binary = iterable_drop(df_binary, threshold)
    track_to_tag, tag_to_track = {}, {}
    for tag in df_binary:
        track_list = list(df_binary[df_binary[tag] == 1].index)
        tag_to_track[tag] = list(set(track_list))

    for idx in range(len(df_binary)):
        instance = df_binary.iloc[idx]
        track_to_tag[instance.name] = list(instance[instance == 1].index)
    return df_binary, track_to_tag, tag_to_track


def song_describer_tag_processor(annotations, tokenizer, threshold=10):
    """
    song_describer, get tag_based_rerieval groundturth
    """
    aspect_list = annotations["aspect_list"]
    tag_list = list(set(flatten_list(aspect_list)))
    normalize_tag = [normalize_text(tag) for tag in tag_list]
    tokens = tokenizer.batch_encode_plus(normalize_tag, add_special_tokens=False)[
        "input_ids"
    ]
    tag_map = _generate_tag_map(normalize_tag, tokens)
    aspect_list = _apply_tag_map(aspect_list, tag_map)
    df_binary, track_to_tag, tag_to_track = _get_retrieval_ground_turth(
        aspect_list=aspect_list,
        annotations=annotations,
        idx_col="track_id",
        threshold=threshold,
    )
    return df_binary, track_to_tag, tag_to_track, tag_map

def music_caps_tag_processor(annotations, tokenizer, threshold=10):
    """
    music_caps, get tag_based_rerieval groundturth
    """
    annotations = annotations[annotations["is_audioset_eval"]]
    aspect_list = annotations["aspect_list"]
    tag_list = list(set(flatten_list(aspect_list)))
    normalize_tag = [normalize_text(tag) for tag in tag_list]
    tokens = tokenizer.batch_encode_plus(normalize_tag, add_special_tokens=False)[
        "input_ids"
    ]
    tag_map = _generate_tag_map(normalize_tag, tokens)
    aspect_list = _apply_tag_map(aspect_list, tag_map)
    df_binary, track_to_tag, tag_to_track = _get_retrieval_ground_turth(
        aspect_list=aspect_list,
        annotations=annotations,
        idx_col="ytid",
        threshold=threshold,
    )
    return df_binary, track_to_tag, tag_to_track, tag_map
