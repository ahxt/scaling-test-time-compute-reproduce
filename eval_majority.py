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


# from latex2sympy2 import latex2sympy
# from sympy import latex, simplify

from utils_qwen import extract_answer, strip_string, math_equal




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./outputs_sgl_parsed")
    args = parser.parse_args()



    majorty = [1,  2,  3,  4,  5,  6,  7,  8,  16,  32,  64, 96, 128, 192, 256, 368, 512, 768, 1024]
    repeat =  [10, 10, 10, 10, 10, 10, 10, 10, 10,  10,  10, 10, 10,  10,  10,  10,  10,  10,  1] 





    all_data = []
    for filename in os.listdir(args.input_dir):
        with open(os.path.join(args.input_dir, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)
        all_data.append(data)



    for repeat_num, m in zip(repeat, majorty):
        m_accs = []
        for repeat_i in range(repeat_num):
            # print("m", m, "repeat_i", repeat_i)
            majority_voting_num = 0
            for data in all_data:
                answer = data["answer"]
                generated_answers = data["generated_answers"]
                
                # Random sampling and filtering
                random_sample = random.sample(generated_answers, m)
                # random_sample = [x for x in random_sample if x is not None]
                most_common_answer = Counter(random_sample).most_common(1)[0][0]

                if most_common_answer in ["None"]:
                    try:
                        most_common_answer = Counter(random_sample).most_common(2)[1][0]
                    except:
                        most_common_answer = ""

                # Check equality and increment count
                if math_equal(most_common_answer, answer):
                    majority_voting_num += 1 

            # Store accuracy for each repeat
            m_accs.append(majority_voting_num / len(all_data))

        # Print statistics
        # print(m, m_accs, np.mean(m_accs), np.std(m_accs))
        # print(f"{args.input_dir}, greedy, {correct_num / 500}")
        print(f"##########{args.input_dir}, majority, {m_accs}, {np.mean(m_accs)}, {np.std(m_accs)}")




