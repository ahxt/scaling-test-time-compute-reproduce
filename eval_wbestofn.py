import json
import os
import random
from collections import Counter
import numpy as np

import sys,os,re
import json
import random
from collections import defaultdict  
from typing import List           
from datasets import load_dataset
import json
import os
import random
import numpy as np
import argparse
import pandas as pd


# from latex2sympy2 import latex2sympy
# from sympy import latex, simplify

from utils_qwen import extract_answer, strip_string, math_equal




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./tmp")
    args = parser.parse_args()



    all_data = []
    for filename in os.listdir(args.input_dir):
        with open(os.path.join(args.input_dir, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)
        all_data.append(data)

    majorty = [1,  2,  3,  4,  5,  6,  7,  8,  16,  32,  64, 96, 128, 192, 256, 368, 512, 768, 1024]
    repeat =  [10, 10, 10, 10, 10, 10, 10, 10, 10,  10,  10, 10, 10,  10,  10,  10,  10,  10,  1] 

    for repeat_num, m in zip(repeat, majorty):
        m_accs = []
        for j in range(repeat_num):
            correct = 0
            for data in all_data:
                generated_answers = data["generated_answers"]
                prm_scores = data["prm_scores"]

                zip_list = list(zip(generated_answers, prm_scores))
                zip_list = random.sample(zip_list, m)

                # Example input
                # zip_list = [("key1", 0.8), ("key2", 0.6), ("key1", 0.9), ("key3", 0.7), ("key2", 0.4)]
                df = pd.DataFrame(zip_list, columns=["key", "score"])
                avg_scores = df.groupby('key', as_index=False)['score'].sum()
                zip_list = list(avg_scores.itertuples(index=False, name=None))

                # print("zip_list", zip_list)
                ranked_list = sorted(zip_list, key=lambda x: x[1], reverse=True)

                # print("ranked_list", ranked_list[0])

                if ranked_list[0][0] == data["answer"]:
                    correct += 1
            m_accs.append(correct / 500)

        print(f"##########{args.input_dir}, wbestofn, {m_accs}, {np.mean(m_accs)}, {np.std(m_accs)}")






