import re
import ast
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

flatten_list = lambda lst: [item for sublist in lst for item in sublist]

def group_to_list(group):
    return list(group)

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
        sort_list = list(set(norm_list))
        norm_lists.append(", ".join(sort_list))
    return norm_lists

def _get_ground_turth(annotations, idx_col, multitag_col):
    df_grouped = annotations.groupby(idx_col)[multitag_col].apply(group_to_list)
    mlb = MultiLabelBinarizer()
    binarys = mlb.fit_transform(list(df_grouped))
    df_binary = pd.DataFrame(
        binarys, index=list(df_grouped.index), columns=mlb.classes_
    )
    track_to_multitag, multitag_to_track = {}, {}
    for multitag in df_binary:
        track_list = list(df_binary[df_binary[multitag] == 1].index)
        multitag_to_track[multitag] = list(set(track_list))
    
    for idx in range(len(df_binary)):
        instance = df_binary.iloc[idx]
        track_to_multitag[instance.name] = list(instance[instance == 1].index)
    return df_binary, track_to_multitag, multitag_to_track

def song_describer_multitag_processor(annotations, tokenizer):
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
    annotations["multi_tag"] = aspect_list

    df_binary, track_to_multitag, multitag_to_track = _get_ground_turth(
        annotations = annotations, 
        idx_col = "track_id", 
        multitag_col = "multi_tag"
    )
    return df_binary, track_to_multitag, multitag_to_track


def music_caps_multitag_processor(annotations, tokenizer):
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
    annotations["multi_tag"] = aspect_list
    
    df_binary, track_to_multitag, multitag_to_track = _get_ground_turth(
        annotations = annotations, 
        idx_col = "ytid", 
        multitag_col = "multi_tag"
    )
    return df_binary, track_to_multitag, multitag_to_track
