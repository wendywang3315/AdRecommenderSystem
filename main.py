from pathlib import Path
import pickle
from preprocess import load_data
from train import train

if __name__ == "__main__":
    # Ignore line 77
    caption_bert_path = Path("./data/caption_bert.data")
    score_data = Path("./data/Augmented-ads-16.csv")

    file = open(caption_bert_path, 'rb')
    captions_bert = pickle.load(file)

    embed_dim = captions_bert.shape[1]
    output_class = 5
    kernel_num = 100
    kernel_size = (3, 4, 5)

    scores = load_data(score_data)

    #train(captions_bert, scores, 10, 32)

