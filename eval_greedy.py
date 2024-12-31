import json
import os
import argparse
from utils_qwen import math_equal




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./tmp")
    args = parser.parse_args()


    all_data = []
    for filename in os.listdir(args.input_dir):
        with open(os.path.join(args.input_dir, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)
        all_data.append(data)


    correct_num = 0
    for data in all_data:
        answer = data["answer"]
        generated_answer = data["generated_answers"][0]
    

        # Check equality and increment count
        if math_equal(generated_answer, answer):
            correct_num += 1 

    print(f"##########{args.input_dir}, greedy, {correct_num / 500}")




