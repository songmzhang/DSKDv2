from typing import Iterable, Dict
import gzip
import json, jsonlines
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from openai import OpenAI



def generate_GIT_completion(model, client, messages):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        top_p=0.8,
        max_tokens=4096,
        seed=10
    )
    # print(completion)
    response = completion.choices[0].message.content

    return response


def generate_MATH_completion(model, client, instruction):
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=2048
    )
    response = completion.choices[0].message.content

    return response


def generate_CODE_completion(model, client, instruction):
    system_prompt = "You are an excellent assistant! Please follow the description to generate the code function."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        extra_body={
            "repetition_penalty": 1.05,
        },
        max_tokens=4096
    )
    response = completion.choices[0].message.content

    return response


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, 
                        default="")
    parser.add_argument("--dataset", type=str, 
                        default="GIT")
    parser.add_argument("--ip", type=str)
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--part", type=int, default=None)
    parser.add_argument("--save_path", type=str, 
                        default="GIT")
    args = parser.parse_args()

    print(args)

    save_path = args.save_path
    if args.dataset == "GIT":
        data_path = "path/to/ultrachat-sampled50k/train.jsonl"
        problems = []
        with jsonlines.open(data_path) as fr:
            for i in fr:
                problems.append(i)
    elif args.dataset == "MATH":
        data_path = "/path/to/metamathQA-sampled50k/train.json"
        problems = json.load(open(data_path))
    elif args.dataset == "CODE":
        data_path = "/path/to/magicoder-sampled10k/train.json"
        problems = json.load(open(data_path))
    else:
        print("error")
        hhhh
    
    model_name = args.model_name
    print(model_name)
    print(len(problems))

    if not os.path.exists(save_path): os.makedirs(save_path)

    openai_api_key = "EMPTY"
    openai_api_base = args.ip

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    print(client)
    
    if args.split is not None:
        assert args.part is not None
        cnt = int(len(problems) / args.split) + 1
        st = args.part * cnt
        ed = min((args.part + 1) * cnt, len(problems))
        print(st, ed)
        cur_problems = problems[st:ed]
        save_f = open(f"{save_path}/{args.dataset}-p{args.part}.json", 'w', encoding="utf-8")
    else:
        cur_problems = problems
        save_f = open(f"{save_path}/{args.dataset}.json", 'w', encoding="utf-8")
        
    save_res = []
    for i_data in tqdm(range(len(cur_problems))):
        data = cur_problems[i_data]
        # print(data)
        cur_save_data = data
        if args.dataset == "GIT":
            cur_messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Please answer my question."},
                ]
            for i, i_item in enumerate(data["messages"]):
                if i % 2 == 0: assert i_item["role"] == "user" and i_item["content"] != ""
                if i % 2 == 1: assert i_item["role"] == "assistant" and i_item["content"] != ""
                if i == len(data["messages"])-1: 
                    assert i_item["role"] == "assistant" and i_item["content"] != ""
                    break
                cur_messages.append({"role": i_item["role"], "content": i_item["content"]})
            # print(cur_messages)
            cur_gen_response = generate_GIT_completion(model_name, client, cur_messages)
            # print(cur_gen_response)
            assert cur_save_data["messages"][-1]["role"] == "assistant"
            cur_save_data["messages"][-1]["content"] = cur_gen_response
        elif args.dataset == "MATH":
            cur_gen_response = generate_MATH_completion(model_name, client, data["query"])
            cur_save_data["response"] = cur_gen_response
        elif args.dataset == "CODE":
            cur_gen_response = generate_CODE_completion(model_name, client, data["instruction"])
            cur_save_data["answer"] = cur_gen_response
            print(cur_gen_response)

        save_res.append(cur_save_data)
    json.dump(save_res, save_f, ensure_ascii=False, indent=4)
    