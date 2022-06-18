import os
import re
from utils import WordTable

data_dir = "../data"

def getCorpus(text_raw):
    '''预处理函数，将原始文本处理为断句断好的列表'''
    # re_preprocess = re.compile('[a-zA-Z0-9’"#$%&\'()*+,-./:：;<=>?@?★、…【】《》？“”‘’！[\\]^_`{|}~]+')
    re_preprocess = re.compile('[a-zA-Z0-9’"#$%&\'()（）*+,-./:：;<=>?@?★、〖〗【】《》“”‘’[\\]^_`{|}~]+')
    text_raw = re_preprocess.sub("",text_raw)
    punctuationL =["\t", "\n", "\u3000", "\u0020", "\u00A0", "\u001a", " "]
    for i in punctuationL:
        text_raw = text_raw.replace(i,"")
    seqendL = ["？","！","……"]
    for i in seqendL:
        text_raw = text_raw.replace(i,"。")
    # corpus = text_raw.replace("。","\n")
    corpus = text_raw.split("。")
    return corpus

if __name__ == '__main__':
    corpus_list = []
    for data_file in os.listdir(data_dir):
        data_path = os.path.join(data_dir, data_file)
        with open(data_path,"r",encoding="GB18030") as fp:
            for i in range(3):
                fp.readline()
            text_raw = "".join(fp.readlines())    
        corpus = getCorpus(text_raw)
        corpus_list.append(corpus)
    word_table = WordTable()
    word_table.build(corpus_list)
