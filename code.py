"""
Read COCO captions JSON, build tokenizer, create sequences and save:
- tokenizer.json
- captions_data.npz
"""

import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict

# ----------------------------------------------------
# 🔧 CONFIGURATION — YOU MUST SET THESE 5 VALUES
# ----------------------------------------------------
ANNOTATIONS_DIR = r"/path/to/coco/annotations"   # folder containing captions_train2017.json
FEATURE_DIR     = r"/path/to/features"           # folder containing .npy feature files
VOCAB_SIZE      = 10000                          # tokenizer vocabulary size
MAX_LEN         = 50                             # max caption length to pad
SUBSET          = None                           # or e.g., 5000 for faster runs
# ----------------------------------------------------

# Resolve output directory
try:
    OUT_DIR = os.path.dirname(__file__)
except NameError:
    OUT_DIR = os.getcwd()

TOKENIZER_PATH = os.path.join(OUT_DIR, "tokenizer.json")
CAPTION_DATA_PATH = os.path.join(OUT_DIR, "captions_data.npz")

# ----------------------------------------------------
# Load COCO captions JSON
# ----------------------------------------------------
ann_file = None
for fn in os.listdir(ANNOTATIONS_DIR):
    if "captions_train" in fn:
        ann_file = os.path.join(ANNOTATIONS_DIR, fn)
        break

if not ann_file:
    raise FileNotFoundError("captions_train*.json not found in ANNOTATIONS_DIR")

with open(ann_file, "r", encoding="utf-8") as f:
    coco = json.load(f)

# ----------------------------------------------------
# Map image_id → list of cleaned captions
# ----------------------------------------------------
id2caps = defaultdict(list)
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    caption = ann["caption"].strip().lower()
    caption = "<start> " + caption + " <end>"
    id2caps[img_id].append(caption)

# ----------------------------------------------------
# Select only images with corresponding feature .npy file
# ----------------------------------------------------
img_id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}

pairs = []  # (image_id, image_name, caption_text)

for img_id, fname in img_id_to_fname.items():
    feat_file = os.path.splitext(fname)[0] + ".npy"
    feat_path = os.path.join(FEATURE_DIR, feat_file)

    if os.path.exists(feat_path):
        for cap in id2caps.get(img_id, []):
            pairs.append((img_id, fname, cap))

print("Total image-caption pairs with cached features:", len(pairs))

if SUBSET:
    pairs = pairs[:SUBSET]
    print("Using subset:", len(pairs))

# ----------------------------------------------------
# Prepare captions
# ----------------------------------------------------
all_captions = [p[2] for p in pairs]
print("Total captions:", len(all_captions))

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<unk>",
    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n'
)
tokenizer.fit_on_texts(all_captions)

# Save tokenizer
with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())

print("Tokenizer saved to", TOKENIZER_PATH)

# ----------------------------------------------------
# Convert captions → padded sequences
# ----------------------------------------------------
sequences = tokenizer.texts_to_sequences(all_captions)

max_len = min(MAX_LEN, max(len(seq) for seq in sequences))
print("Using max_len =", max_len)

padded = pad_sequences(sequences, maxlen=max_len, padding="post")
img_fnames = [p[1] for p in pairs]

# Save NPZ
np.savez_compressed(
    CAPTION_DATA_PATH,
    padded=padded,
    img_fnames=np.array(img_fnames)
)

print("Caption dataset saved to", CAPTION_DATA_PATH)
