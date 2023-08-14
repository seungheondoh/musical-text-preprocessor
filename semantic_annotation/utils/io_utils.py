import json

def save_jsonl(fname, data):
    with open(fname, encoding= "utf-8",mode="w") as file: 
    	for i in data: file.write(json.dumps(i) + "\n")

def load_jsonl(fname):
    with open(fname, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_idx(list_of_dict, list_of_dict_name):
    for item, fname in zip(list_of_dict, list_of_dict_name):
        with open(f"{fname}.json", mode="w") as io:
            json.dump(item, io, indent=4)