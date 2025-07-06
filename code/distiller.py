import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from utils import log_rank, print_rank


class Distiller(nn.Module):
    def __init__(self, args, device):
        super(Distiller, self).__init__()
        self.args = args
        self.device = device
        self.student_model_type = args.model_type
        self.student_model, self.student_tokenizer = self.load_student_model()
        
        if self.args.teacher_model_path is not None:
            self.teacher_model, self.teacher_tokenizer = self.load_teacher_model()
        else:
            self.teacher_model, self.teacher_tokenizer = None, None

        log_rank(f"student tokenizer: {self.student_tokenizer}")
        log_rank(f"teacher tokenizer: {self.teacher_tokenizer}")
        
        if self.teacher_model and args.projector_config_path:
            self.set_and_load_existing_projectors()
            log_rank(f"projector structure: {self.projectors}")
        
        if args.teacher_to_student_token_mapping is not None:  # only for MinED
            self.tea2stu_token_mapping = json.load(open(args.teacher_to_student_token_mapping))
            log_rank(f"Load teacher-to-student token mapping from {args.teacher_to_student_token_mapping}")
        
        if args.teacher_to_student_id_mapping is not None:  # only for MinED
            self.tea2stu_id_mapping = json.load(open(args.teacher_to_student_id_mapping))
            log_rank(f"Load teacher-to-student id mapping from {args.teacher_to_student_id_mapping}")

            self.stu2tea_id_mapping = {}
            for tea_id in self.tea2stu_id_mapping:
                if self.tea2stu_id_mapping[tea_id] not in self.stu2tea_id_mapping:
                    self.stu2tea_id_mapping[self.tea2stu_id_mapping[tea_id]] = [int(tea_id)]
                else:
                    self.stu2tea_id_mapping[self.tea2stu_id_mapping[tea_id]].append(int(tea_id))
            
            max_align_num = 1
            for stu_id in self.stu2tea_id_mapping:
                self.stu2tea_id_mapping[stu_id] = self.stu2tea_id_mapping[stu_id][:max_align_num] + \
                    [self.stu2tea_id_mapping[stu_id][-1]] \
                        * max(0, max_align_num - len(self.stu2tea_id_mapping[stu_id]))
                
            self.tea2stu_id_mapping = torch.LongTensor(list(self.tea2stu_id_mapping.values())).to(device)
            self.stu2tea_id_mapping_tea = torch.LongTensor(list(self.stu2tea_id_mapping.values())).to(device)
            self.stu2tea_id_mapping_stu = torch.LongTensor(list(self.stu2tea_id_mapping.keys())).to(device)

    @staticmethod
    def add_distiller_args(parser):
        group = parser.add_argument_group("distiller", "distiller configurations")
        group.add_argument("--projector-lr", type=float, default=0.001,
                           help='learning rate only for projection')
        group.add_argument("--vocab-alignment-path", type=str, default=None,
                           help='path for the vocab alignment file')
        group.add_argument("--teacher-to-student-token-mapping", type=str, default=None,
                           help='path for the vocab alignment file (token, teacher-to-student)')
        group.add_argument("--teacher-to-student-id-mapping", type=str, default=None,
                           help='path for the vocab alignment file (id, teacher-to-student)')
        group.add_argument("--student-to-teacher-token-mapping", type=str, default=None,
                           help='path for the vocab alignment file (token, student-to-teacher)')
        group.add_argument("--student-to-teacher-id-mapping", type=str, default=None,
                           help='path for the vocab alignment file (id, student-to-teacher)')
        group.add_argument("--init-t2s-projector", action="store_true",
                           help='whether to init t2s projector')
        group.add_argument("--init-s2t-projector", action="store_true",
                           help='whether to init s2t projector')
        group.add_argument("--topk-vocab", type=int, default=-1,
                           help='use embeddings of topk vocabulary for projector initialization')
        group.add_argument("--only-stu-kd", action="store_true",
                           help='only conduct kd in student space')
        group.add_argument("--only-tea-kd", action="store_true",
                           help='only conduct kd in teacher space')
        group.add_argument("--t2s-agreement", type=float, default=1.0,
                           help='whether adds the t2s_agreement_mask')
        
        return parser
    
    def load_tokenizer(self, model_type, path):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if model_type == "qwen":
            tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def set_and_load_existing_projectors(self):
        self.t2s_projector = nn.Linear(self.teacher_hidden_size, self.student_hidden_size, bias=False)
        self.s2t_projector = nn.Linear(self.student_hidden_size, self.teacher_hidden_size, bias=False)
        if self.args.criterion == "dual_space_kd_with_cma":
            self.query_projector = nn.Linear(self.student_hidden_size * 2, self.teacher_hidden_size * 2, bias=False)
        if self.args.init_t2s_projector or self.init_s2t_projector:
            self.projector_init()
            
    def projector_init(self):
        """
        projector initialization aims to achieve logit equivalence (take W^{t2s} as an example): 
        t_logits = t2s_logits
        => H^t * W^t = H^t * W^{t2s} * W^s
        => W^{t2s} = W^t * pseudo_inverse(W^s)
        
        s2t_projector (W^{s2t}) will be initialized in the criterion if args.init_s2t_projector is True
        """
        student_head = self.student_model.lm_head.weight.detach().transpose(0, 1)  
        teacher_head = self.teacher_model.lm_head.weight.transpose(0, 1) 

        if self.student_tokenizer.get_vocab().items() == self.teacher_tokenizer.get_vocab().items():
            if self.args.topk_vocab != -1:     # only use part of vocab to reduce initialization error
                part_student_head = student_head[:, :self.args.topk_vocab]
                part_teacher_head = teacher_head[:, :self.args.topk_vocab]
            else:
                part_student_head = student_head
                part_teacher_head = teacher_head
        else:  # different vocab: only use the overlapped part of both vocabularies
            student_vocab = {k.replace("Ġ", "▁"): v for k, v in self.student_tokenizer.get_vocab().items()}
            teacher_vocab = {k.replace("Ġ", "▁"): v for k, v in self.teacher_tokenizer.get_vocab().items()}
            overlap_tokens = [k for k in student_vocab if k in teacher_vocab]
            log_rank(f"Found overlap tokens of two tokenizers: {len(overlap_tokens)}")
            student_overlap_token_ids = torch.tensor([student_vocab[token] for token in overlap_tokens], dtype=torch.long, device=student_head.device)
            teacher_overlap_token_ids = torch.tensor([teacher_vocab[token] for token in overlap_tokens], dtype=torch.long, device=teacher_head.device)
            part_student_head = student_head[:, student_overlap_token_ids]
            part_teacher_head = teacher_head[:, teacher_overlap_token_ids]
            if self.args.topk_vocab != -1:
                part_student_head = part_student_head[:, :self.args.topk_vocab]
                part_teacher_head = part_teacher_head[:, :self.args.topk_vocab]

        if self.args.init_t2s_projector:
            log_rank("Init t2s projector through pseudo inverse")
            part_student_head_pinv = torch.linalg.pinv(part_student_head.float())
            init_t2s = (part_teacher_head.float() @ part_student_head_pinv).transpose(0, 1)
            self.t2s_projector.weight.data.copy_(init_t2s.to(student_head))
            if hasattr(self.t2s_projector, "bias") and self.t2s_projector.bias is not None:
                self.t2s_projector.bias.data.copy_(torch.zeros_like(self.t2s_projector.bias))
            log_rank("Projector Initialization Finished")

        if self.args.init_s2t_projector:
            self.part_teacher_head_pinv = torch.linalg.pinv(part_teacher_head.float())
            self.part_teacher_head_pinv.requires_grad = False

    def load_student_model(self):
        config = AutoConfig.from_pretrained(self.args.model_path, trust_remote_code=True)
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer(self.args.model_type, self.args.model_path)
        
        if hasattr(config, "n_embed"):
            self.student_hidden_size = config.n_embed
        else:
            self.student_hidden_size = config.hidden_size

        if self.args.model_dtype == "fp32":
            self.dtype = torch.float32
        elif self.args.model_dtype == "bf16":
            self.dtype = torch.bfloat16
        elif self.args.model_dtype == "fp16":
            self.dtype = torch.float16
        else:
            raise NotImplementedError("Invalid model_dtype for f`{self.args.model_dtype}`")
        log_rank(f"Loading student model with {self.args.model_dtype}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path, 
            config=config, 
            device_map={"": self.device}, 
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
            
        if self.args.peft is not None:
            if self.args.peft == "lora":
                model.enable_input_require_grads()
                if self.args.peft_path is not None:
                    if self.args.do_train:
                        _model = PeftModel.from_pretrained(model, self.args.peft_path)
                        state_dict = dict(_model.state_dict().items())
                        peft_config = LoraConfig(
                            task_type=TaskType.CAUSAL_LM, 
                            inference_mode=(not self.args.do_train), 
                            r=self.args.peft_lora_r, 
                            lora_alpha=self.args.peft_lora_alpha, 
                            lora_dropout=self.args.peft_lora_dropout
                        )
                        model = get_peft_model(model, peft_config)
                        model.load_state_dict(state_dict)
                        del _model
                        del state_dict
                    else:
                        model = PeftModel.from_pretrained(model, self.args.peft_path)
                else:
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM, 
                        inference_mode=(not self.args.do_train), 
                        r=self.args.peft_lora_r, 
                        lora_alpha=self.args.peft_lora_alpha, 
                        lora_dropout=self.args.peft_lora_dropout
                    )
                    model = get_peft_model(model, peft_config)
            else:
                raise NotImplementedError
        
        trainable_params, all_param = self.count_trainable_parameters(model)
        log_rank(
            "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
                trainable_params, all_param, 100 * trainable_params / all_param
            )
        )
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model, tokenizer
    
    def count_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param
    
    def load_teacher_model(self):
        config = AutoConfig.from_pretrained(self.args.teacher_model_path)
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer(self.args.teacher_model_type, self.args.teacher_model_path)

        if hasattr(config, "n_embed"):
            self.teacher_hidden_size = config.n_embed
        else:
            self.teacher_hidden_size = config.hidden_size

        log_rank(f"Loading teacher model with {self.args.model_dtype}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.args.teacher_model_path, 
            config=config, 
            device_map={"": self.device}, 
            torch_dtype=self.dtype
        )

        if self.args.peft is not None and self.args.teacher_peft_path is not None:
            if self.args.peft == "lora":
                model = PeftModel.from_pretrained(model, self.args.teacher_peft_path)
                model = model.merge_and_unload()
            else:
                raise NotImplementedError
        else:
            log_rank(' > number of parameters of the teacher model: {:,}'.format(
                sum([p.nelement() for p in model.parameters()])
            ))
        for params in model.parameters():
            params.requires_grad = False
        return model, tokenizer
    
    def add_optimizer_param_group(self, optimizer):
        if hasattr(self, "t2s_projector"):
            if self.args.projector_lr:
                optimizer.add_param_group({
                    "params": [p for p in self.t2s_projector.parameters()],
                    "lr": self.args.projector_lr
                })
            else:
                optimizer.add_param_group({
                    "params": [p for p in self.t2s_projector.parameters()],
                })
        return optimizer

    def forward(self, criterion, batch, logging_output):
        loss, logging_output = criterion(self, batch, logging_output)
        return loss, logging_output

    def student_generate(self, prompt_batch, generation_config, max_new_tokens):
        outputs = self.student_model.generate(
            **prompt_batch, 
            generation_config=generation_config, 
            max_new_tokens=max_new_tokens
        )
        return outputs
    
    def teacher_generate(self, prompt_batch, generation_config, max_new_tokens):
        outputs = self.teacher_model.generate(
            **prompt_batch, 
            generation_config=generation_config, 
            max_new_tokens=max_new_tokens
        )
        return outputs
    
    def mixed_generate(self, prompt_batch, generation_config, max_new_tokens):
        """ Only supported by the modified transformers provided by Gu et al. (2024). 
        Install with `pip3 install git+https://github.com/t1101675/transformers@minillm` 
        """
        outputs = self.student_model.generate(
            **prompt_batch, 
            generation_config=generation_config, 
            max_new_tokens=max_new_tokens,
            mix_in_model=self.teacher_model,
            mix_in_alpha=0.2
        )
        return outputs