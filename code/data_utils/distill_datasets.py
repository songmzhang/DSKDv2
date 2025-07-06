import torch
import os
import json, jsonlines
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm

from utils import log_rank
from typing import Dict, Optional
from transformers import AutoTokenizer


class DistillDataset(Dataset):
    def __init__(
        self, 
        args, 
        split: str,
        student_tokenizer: AutoTokenizer, 
        teacher_tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.args = args
        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.dataset = self._load_and_process_data()
        # log_rank(f"Num of data instances: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, index):
        return self.dataset[index]
    
    def _load_and_process_data(self):
        dataset = []
        if "math" in self.args.data_dir.lower() or "code" in self.args.data_dir.lower():
            path = os.path.join(self.args.data_dir, f"{self.split}.json")
        else:
            path = os.path.join(self.args.data_dir, f"{self.split}.jsonl")
        if os.path.exists(path):
            if "math" in path.lower():
                prompt_length = []
                response_length = []

                raw_data = json.load(open(path))
                log_rank(f"Processing Math dataset {len(raw_data)} for student model (and teacher model)...")  
                seg = np.iinfo(np.int32).max * 2 + 1        
                for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):
                    cur_messages = [
                        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                        {"role": "user", "content": data["query"]}, 
                    ]
                    cur_prompt_stu = self.student_tokenizer.apply_chat_template(
                        cur_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    student_prompt_ids = self.student_tokenizer.encode(
                        cur_prompt_stu, add_special_tokens=False
                    )
                    prompt_length.append(len(student_prompt_ids))
                    student_prompt_ids = student_prompt_ids[:self.max_prompt_length]
                    student_response_ids = self.student_tokenizer.encode(
                        data["response"], add_special_tokens=False
                    )
                    student_response_ids = student_response_ids \
                                        + [self.student_tokenizer.eos_token_id]
                    response_length.append(len(student_response_ids))
                    tokenized_data = {
                        "student_input_ids": student_prompt_ids + [seg] + student_response_ids,
                    }

                    if self.teacher_tokenizer is not None:         
                        cur_prompt_tea = self.teacher_tokenizer.apply_chat_template(
                            cur_messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )   
                        teacher_prompt_ids = self.teacher_tokenizer.encode(
                            cur_prompt_tea, add_special_tokens=False
                        )
                        teacher_prompt_ids = teacher_prompt_ids[:self.max_prompt_length]
                        teacher_response_ids = self.teacher_tokenizer.encode(
                            data["response"], add_special_tokens=False
                        )
                        teacher_response_ids = teacher_response_ids \
                                                + [self.teacher_tokenizer.eos_token_id]
                        tokenized_data[f"teacher_input_ids"] = \
                            teacher_prompt_ids + [seg] + teacher_response_ids

                    dataset.append(tokenized_data)

                print("prompt_length (min/max/mean):", min(prompt_length), max(prompt_length), np.mean(prompt_length))
                print("response_length (min/max/mean): ", min(response_length), max(response_length), np.mean(response_length))
                cnt = 0
                for i in prompt_length:
                    if i > 350:
                        cnt += 1
                print("prompt_length > 350:", cnt, len(dataset))
                cnt = 0
                for i, j in zip(prompt_length, response_length):
                    if i+j > 1024:
                        cnt += 1
                print("prompt_length + response_length > 1024:", cnt, len(dataset))
            elif "code" in path.lower():
                prompt_length = []
                response_length = []

                raw_data = json.load(open(path))
                log_rank(f"Processing Math dataset {len(raw_data)} for student model (and teacher model)...")  
                seg = np.iinfo(np.int32).max * 2 + 1        
                for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):
                    cur_messages = [
                        {"role": "system", "content": "You are an excellent assistant! Please follow the description to generate the code function."},
                        {"role": "user", "content": data["instruction"]}, 
                    ]
                    cur_prompt_stu = self.student_tokenizer.apply_chat_template(
                        cur_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    student_prompt_ids = self.student_tokenizer.encode(
                        cur_prompt_stu, add_special_tokens=False
                    )
                    prompt_length.append(len(student_prompt_ids))
                    student_prompt_ids = student_prompt_ids[:self.max_prompt_length]
                    student_response_ids = self.student_tokenizer.encode(
                        data["answer"], add_special_tokens=False
                    )
                    student_response_ids = student_response_ids \
                                        + [self.student_tokenizer.eos_token_id]
                    response_length.append(len(student_response_ids))
                    tokenized_data = {
                        "student_input_ids": student_prompt_ids + [seg] + student_response_ids,
                    }

                    if self.teacher_tokenizer is not None:         
                        cur_prompt_tea = self.teacher_tokenizer.apply_chat_template(
                            cur_messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )   
                        teacher_prompt_ids = self.teacher_tokenizer.encode(
                            cur_prompt_tea, add_special_tokens=False
                        )
                        teacher_prompt_ids = teacher_prompt_ids[:self.max_prompt_length]
                        teacher_response_ids = self.teacher_tokenizer.encode(
                            data["answer"], add_special_tokens=False
                        )
                        teacher_response_ids = teacher_response_ids \
                                                + [self.teacher_tokenizer.eos_token_id]
                        tokenized_data[f"teacher_input_ids"] = \
                            teacher_prompt_ids + [seg] + teacher_response_ids

                    dataset.append(tokenized_data)

                print("prompt_length (min/max/mean):", min(prompt_length), max(prompt_length), np.mean(prompt_length))
                print("response_length (min/max/mean): ", min(response_length), max(response_length), np.mean(response_length))
                cnt = 0
                for i in prompt_length:
                    if i > 750:
                        cnt += 1
                print("prompt_length > 750:", cnt, len(dataset))
                cnt = 0
                for i, j in zip(prompt_length, response_length):
                    if i+j > 1280:
                        cnt += 1
                print("prompt_length + response_length > 1280:", cnt, len(dataset))
            elif "ultrachat" in path.lower():
                prompt_length = []
                response_length = []

                raw_data = []
                with jsonlines.open(path) as reader1:
                    for item in reader1:
                        raw_data.append(item)
                log_rank(f"Processing Math dataset {len(raw_data)} for student model (and teacher model)...")  
                seg = np.iinfo(np.int32).max * 2 + 1        
                for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):
                    cur_messages = [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Please answer my question."},
                    ]
                    for i, i_item in enumerate(data["messages"]):
                        if i % 2 == 0: assert i_item["role"] == "user" and i_item["content"] != ""
                        if i % 2 == 1: assert i_item["role"] == "assistant" and i_item["content"] != ""
                        if i == len(data["messages"])-1: 
                            assert i_item["role"] == "assistant" and i_item["content"] != ""
                            cur_response = i_item["content"]
                            break
                        cur_messages.append({"role": i_item["role"], "content": i_item["content"]})

                    cur_prompt_stu = self.student_tokenizer.apply_chat_template(
                        cur_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    student_prompt_ids = self.student_tokenizer.encode(
                        cur_prompt_stu, add_special_tokens=False
                    )
                    prompt_length.append(len(student_prompt_ids))
                    student_prompt_ids = student_prompt_ids[:self.max_prompt_length]
                    student_response_ids = self.student_tokenizer.encode(
                        cur_response, add_special_tokens=False
                    )
                    student_response_ids = student_response_ids \
                                        + [self.student_tokenizer.eos_token_id]
                    response_length.append(len(student_response_ids))
                    tokenized_data = {
                        "student_input_ids": student_prompt_ids + [seg] + student_response_ids,
                    }

                    if self.teacher_tokenizer is not None:    
                        cur_prompt_tea = self.teacher_tokenizer.apply_chat_template(
                            cur_messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )     
                        teacher_prompt_ids = self.teacher_tokenizer.encode(
                            cur_prompt_tea, add_special_tokens=False
                        )
                        teacher_prompt_ids = teacher_prompt_ids[:self.max_prompt_length]
                        teacher_response_ids = self.teacher_tokenizer.encode(
                            cur_response, add_special_tokens=False
                        )
                        teacher_response_ids = teacher_response_ids \
                                                + [self.teacher_tokenizer.eos_token_id]
                        tokenized_data[f"teacher_input_ids"] = \
                            teacher_prompt_ids + [seg] + teacher_response_ids
                    
                    dataset.append(tokenized_data)
                print("prompt_length (min/max/mean):", min(prompt_length), max(prompt_length), np.mean(prompt_length))
                print("response_length (min/max/mean): ", min(response_length), max(response_length), np.mean(response_length))
                cnt = 0
                for i in prompt_length:
                    if i > 1500:
                        cnt += 1
                print("prompt_length > 1500:", cnt, len(dataset))
                cnt = 0
                for i, j in zip(prompt_length, response_length):
                    if i+j > 2048:
                        cnt += 1
                print("prompt_length + response_length > 2048:", cnt, len(dataset))
            else:
                with open(path) as f:
                    raw_data = [json.loads(l) for l in f.readlines()]
                    self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in raw_data]
                
                log_rank("Processing dataset for student model (and teacher model)...")  
                seg = np.iinfo(np.int32).max * 2 + 1        
                for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):
                    student_prompt_ids = self.student_tokenizer.encode(
                        data["prompt"], add_special_tokens=False
                    )
                    student_prompt_ids = student_prompt_ids[:self.max_prompt_length]
                    student_response_ids = self.student_tokenizer.encode(
                        data["output"], add_special_tokens=False
                    )
                    student_response_ids = student_response_ids \
                                        + [self.student_tokenizer.eos_token_id]
                    tokenized_data = {
                        "student_input_ids": student_prompt_ids + [seg] + student_response_ids,
                    }

                    if self.teacher_tokenizer is not None:
                    # for model_type in self.teacher_tokenizers:
                    #     if self.teacher_tokenizers[model_type] is None: continue
                            
                        teacher_prompt_ids = self.teacher_tokenizer.encode(
                            data["prompt"], add_special_tokens=False
                        )
                        teacher_prompt_ids = teacher_prompt_ids[:self.max_prompt_length]
                        teacher_response_ids = self.teacher_tokenizer.encode(
                            data["output"], add_special_tokens=False
                        )
                        teacher_response_ids = teacher_response_ids \
                                                + [self.teacher_tokenizer.eos_token_id]
                        tokenized_data[f"teacher_input_ids"] = \
                            teacher_prompt_ids + [seg] + teacher_response_ids

                    dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")
        
    def _process_lm(
        self, 
        i, 
        sample, 
        input_batch, 
        label_batch, 
        prompt_batch, 
        teacher_input_batch, 
        teacher_label_batch, 
        teacher_prompt_batch
    ):
        """Process each data item"""
        seg = np.iinfo(np.int32).max * 2 + 1
        input_ids = np.array(sample["student_input_ids"])
        prompt_len = np.where(input_ids == seg)[0][0]
        prompt = input_ids[:prompt_len]
        input_ids = np.concatenate(
            [input_ids[:prompt_len], input_ids[prompt_len+1:]], 
            axis=0
        )
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        
        input_batch["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        input_batch["attention_mask"][i][:input_len-1] = 1
        if "position_ids" in input_batch:
            input_batch["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
            
        label_batch["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        label_batch["label"][i][:prompt_len-1] = -100
        label_batch["loss_mask"][i][:input_len-1] = 1.0
        label_batch["loss_mask"][i][:prompt_len-1] = 0
        
        prompt_batch["input_ids"][i][-prompt_len:] = torch.tensor(prompt, dtype=torch.long)
        prompt_batch["attention_mask"][i][-prompt_len:] = 1
        if "position_ids" in prompt_batch:
            prompt_batch["position_ids"][i][-prompt_len:] = torch.arange(0, prompt_len, dtype=torch.long)

        if teacher_input_batch is not None:
            t_input_ids = np.array(sample[f"teacher_input_ids"])
            t_prompt_len = np.where(t_input_ids == seg)[0][0]
            t_prompt = t_input_ids[:t_prompt_len]
            t_input_ids = np.concatenate(
                [t_input_ids[:t_prompt_len], t_input_ids[t_prompt_len+1:]], 
                axis=0
            )
            t_input_ids = t_input_ids[:self.max_length]
            t_input_len = len(t_input_ids)
            
            teacher_input_batch["input_ids"][i][:t_input_len-1] = torch.tensor(t_input_ids[:-1], dtype=torch.long)
            teacher_input_batch["attention_mask"][i][:t_input_len-1] = 1
            if "position_ids" in teacher_input_batch:
                teacher_input_batch["position_ids"][i][:t_input_len-1] = torch.arange(0, t_input_len-1, dtype=torch.long)
            
            teacher_label_batch["label"][i][:t_input_len-1] = torch.tensor(t_input_ids[1:], dtype=torch.long)
            teacher_label_batch["label"][i][:t_prompt_len-1] = -100
            teacher_label_batch["loss_mask"][i][:t_input_len-1] = 1.0
            teacher_label_batch["loss_mask"][i][:t_prompt_len-1] = 0
            
            teacher_prompt_batch["input_ids"][i][-t_prompt_len:] = torch.tensor(t_prompt, dtype=torch.long)
            teacher_prompt_batch["attention_mask"][i][-t_prompt_len:] = 1
            if "position_ids" in teacher_prompt_batch:
                teacher_prompt_batch["position_ids"][i][-t_prompt_len:] = torch.arange(0, t_prompt_len, dtype=torch.long)

    @staticmethod
    def move_to_device(datazip, device):
        for data in datazip:
            if isinstance(datazip[data], dict):
                __class__.move_to_device(datazip[data], device)
            else:
                datazip[data] = datazip[data].to(device)
        # for data in datazip:
        #     for k in datazip[data]:
        #         if isinstance(datazip[data][k], torch.Tensor):
        #             datazip[data][k] = datazip[data][k].to(device)
        #         elif isinstance(datazip[data][k], dict):
        #             for kk in datazip[data][k]:
        #                 datazip[data][k][kk] = datazip[data][k][kk].to(device)

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length
        
        stu_pad_id = self.student_tokenizer.pad_token_id
        input_batch = {
            "input_ids": torch.full((bs, max_length), fill_value=stu_pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(bs, max_length, dtype=torch.long),
        }
        if self.args.model_type in ["gpt2"]:     # prepare absolute positional encoding for gpt2
            input_batch["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        label_batch = {
            "label": torch.full((bs, max_length), fill_value=-100, dtype=torch.long),
            "loss_mask": torch.zeros(bs, max_length)
        }
        
        prompt_batch = {
            "input_ids": torch.full((bs, self.max_prompt_length), fill_value=stu_pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }
        # if self.args.model_type in ["gpt2"]:
        #     gen_data["position_ids"] = torch.zeros(bs, self.max_prompt_length, dtype=torch.long)
        
        if self.teacher_tokenizer is not None:
            tea_pad_id = self.teacher_tokenizer.pad_token_id
            teacher_input_batch = {
                "input_ids": torch.full((bs, max_length), fill_value=tea_pad_id, dtype=torch.long),
                "attention_mask": torch.zeros(bs, max_length, dtype=torch.long),
            }
            if self.args.teacher_model_type in ["gpt2"]:
                teacher_input_batch["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
                
            teacher_label_batch = {
                "label": torch.full((bs, max_length), fill_value=-100, dtype=torch.long),
                "loss_mask": torch.zeros(bs, max_length),
            }
            
            teacher_prompt_batch = {
                "input_ids": torch.full((bs, self.max_prompt_length), fill_value=tea_pad_id, dtype=torch.long),
                "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
            }
            # if self.args.teacher_model_type in ["gpt2"]:
            #     teacher_gen_data["position_ids"] = torch.zeros(bs, self.max_prompt_length, dtype=torch.long)
        else:
            teacher_input_batch, teacher_label_batch, teacher_prompt_batch = None, None, None

        for i, sample in enumerate(samples):
            self._process_lm(
                i, sample, input_batch, label_batch, prompt_batch, 
                teacher_input_batch, teacher_label_batch, teacher_prompt_batch
            )
        
        if teacher_input_batch is not None:
            whole_batch = {
                "input_batch": input_batch,
                "label_batch": label_batch,
                "prompt_batch": prompt_batch,
                "teacher_input_batch": teacher_input_batch,
                "teacher_label_batch": teacher_label_batch,
                "teacher_prompt_batch": teacher_prompt_batch,
            }
        else:
            whole_batch = {
                "input_batch": input_batch,
                "label_batch": label_batch,
                "prompt_batch": prompt_batch,
            }
        
        return whole_batch
