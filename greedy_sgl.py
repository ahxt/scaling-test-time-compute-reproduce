import os
import json
import argparse
from datasets import load_dataset
from datetime import datetime
from transformers import AutoTokenizer
import sglang as sgl


system_prompt: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
custom_chat_template: str = '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--system_prompt", type=str, default=system_prompt)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default="./outputs_sgl_4096")
    parser.add_argument("--use_custom_chat_template", type=bool, default=False)
    args = parser.parse_args()
    print(args)

    os.makedirs(args.output_dir, exist_ok=True)

    
    dataset = load_dataset(args.dataset_name, split='test')
    problems = [row["problem"] for row in dataset]


    llm = sgl.Engine(model_path=args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.use_custom_chat_template:
        tokenizer.chat_template = custom_chat_template


    sampling_params = {"temperature": args.temperature, 
                       "top_p": args.top_p, 
                       "top_k": args.top_k, 
                       "max_new_tokens": args.max_new_tokens,
                       "n": args.num_samples}

    

    # Iterate over prompts with progress bar
    for idx, problem in enumerate(problems):
        message = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": problem}
        ]
        prompts = [tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)]
        outputs = llm.generate(prompts, sampling_params)

        responses = [
            {"role": "assistant", "content": output["text"].strip()}
            for output in outputs
        ]

        data = {
            "test_id": idx+1,
            "problem": problem,
            "subject": dataset[idx]["subject"],
            "level": dataset[idx]["level"],
            "answer": dataset[idx]["answer"],
            "prompts": prompts,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "greedy_responses": responses,
            
        }

        # Define individual JSON filename
        individual_filename = f"test_{idx+1}.json"
        individual_path = os.path.join(args.output_dir, individual_filename)

        # Save the individual JSON file
        with open(individual_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"{timestamp},  save {individual_filename}")

    print(f"Generation completed. All data saved to '{args.output_dir}'.")

