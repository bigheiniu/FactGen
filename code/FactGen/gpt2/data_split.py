import pandas as pd
import os

def src_tgt_split(file):
    data = pd.read_csv(file, sep="\t" if "tsv" in file else ",")
    data['src'] = data[['fact_col','title']].agg(" ".join, axis=1)
    data['tgt'] = data['content']
    src = data['src'].values.tolist()
    tgt = data['tgt'].values.tolist()
    file = file.replace("_fact.tsv", ".txt")
    with open(file+".src", 'w') as f1:
        for line in src:
            # line = line.replace("|", " ")
            f1.write(line.replace("\n", " ") + "\n")
    with open(file + ".tgt", 'w') as f1:
        for line in tgt:
            f1.write(line.replace("\n", " ") + "\n")
        
        
if __name__ == '__main__':
    dir = "../../data/news_corpus/gossip"
    for type in ['train','val','test']:
        file_name = os.path.join(dir, type+"_fact.tsv")
        print(file_name)
        print("hello")
        src_tgt_split(file_name)
    