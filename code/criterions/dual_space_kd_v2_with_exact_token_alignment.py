import math
import torch
from .various_divergence import VariousDivergence
from utils import log_rank
import re


def align_sequences(tea_seq, stu_seq, student_tokenizer, teacher_tokenizer):
    i, j = 0, 0
    t2s_align, s2t_align = [], []
    history_tea_seq, history_stu_seq = "", ""
    
    tea_seq = [token.replace('▁', '').replace('Ġ', '') for token in tea_seq]
    stu_seq = [token.replace('▁', '').replace('Ġ', '') for token in stu_seq]

    while i < len(tea_seq) and j < len(stu_seq):
        if history_tea_seq == history_stu_seq and (
            tea_seq[i] == stu_seq[j] or (
                tea_seq[i] == teacher_tokenizer.eos_token and \
                stu_seq[j] == student_tokenizer.eos_token
            )
        ):
            history_tea_seq += tea_seq[i]
            history_stu_seq += stu_seq[j]
            t2s_align.append(i)
            s2t_align.append(j)
            i += 1
            j += 1
        elif len(history_tea_seq) > len(history_stu_seq):
            history_stu_seq += stu_seq[j]
            j += 1
        elif len(history_tea_seq) < len(history_stu_seq):
            history_tea_seq += tea_seq[i]
            i += 1
        else:
            history_tea_seq += tea_seq[i]
            history_stu_seq += stu_seq[j]
            i += 1
            j += 1
            
    return t2s_align, s2t_align


class DualSpaceKDV2WithETA(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)

    def forward(
        self, 
        distiller, 
        batch, 
        logging_output
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        teacher_model.eval()

        self.distiller = distiller
        outputs = model(**batch["input_batch"], output_hidden_states=True)
        logits = outputs.logits
        log = {}
        ce_loss = self.compute_cross_entropy_loss(
            outputs.logits, batch["label_batch"]["label"], reduction="sum"
        )[0] / batch["label_batch"]["loss_denom"]
        log["nll_loss"] = ce_loss

        if "op_input_batch" in batch:     # on-policy scenario
            outputs = model(**batch["op_input_batch"], output_hidden_states=True)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **batch["op_teacher_input_batch"], 
                    output_hidden_states=True
                )
            kd_loss, log = self.compute_on_policy_dual_space_kd_loss_with_eta(
                outputs, teacher_outputs, batch, distiller, log
            )
        else:    # off-policy scenario
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **batch["teacher_input_batch"], 
                    output_hidden_states=True
                )
            kd_loss, log = self.compute_dual_space_kd_loss_with_eta(
                outputs, teacher_outputs, batch, distiller, log
            )
        loss = (1.0 - self.kd_rate) * ce_loss + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(logits, batch["label_batch"])
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, log)
        return loss, logging_output
    

    def compute_dual_space_kd_loss_with_eta(
        self, outputs, teacher_outputs, batch, distiller, log
    ):
        target = batch["label_batch"]["label"]
        teacher_target = batch["teacher_label_batch"]["label"]

        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]
        
        # teacher space
        if self.args.init_s2t_projector:
            stu_lm_head = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
            if self.args.topk_vocab != -1:
                stu_lm_head = stu_lm_head[:, :self.args.topk_vocab]
            s2t_projector = stu_lm_head @ distiller.part_teacher_head_pinv
            s2t_hiddens = hiddens @ s2t_projector
        else:
            stu_v_hiddens = distiller.s2t_projector(hiddens)

        tea_v_hiddens = distiller.t2s_projector(teacher_hiddens)  # m x D x D x d -> m x d

        t_preds = teacher_outputs.logits.argmax(-1)
        t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)

        t_preds_as_label = []
        t2s_hiddens_align = torch.zeros_like(tea_v_hiddens).to(tea_v_hiddens.device).to(tea_v_hiddens)   # m x d -> n x d    在s的维度上是t的hiddens
        s2t_hiddens_align = torch.zeros_like(stu_v_hiddens).to(stu_v_hiddens.device).to(stu_v_hiddens)   # n x D -> m x D    在t的维度上是s的hiddens
        align_ratio = []
        for i in range(t_preds.shape[0]):
            indices_t_preds = torch.where(t_preds[i] != -100)[0]
            indices_t_target = torch.where(teacher_target[i] != -100)[0]
            indices_s_target = torch.where(target[i] != -100)[0]

            if indices_t_preds.shape[0] == 0:
                t_preds_as_label.append(torch.tensor([-100]*t_preds.shape[-1], device=tea_v_hiddens.device))
                align_ratio.append(1.0)
                continue
            
            cur_t_preds = t_preds[i][indices_t_preds[0]: indices_t_preds[-1]+1]
            cur_t_target = teacher_target[i][indices_t_target[0]: indices_t_target[-1]+1]
            cur_t_target_tokens = distiller.teacher_tokenizer.convert_ids_to_tokens(cur_t_target)

            cur_s_target = target[i][indices_s_target[0]: indices_s_target[-1]+1]
            cur_s_target_tokens = distiller.student_tokenizer.convert_ids_to_tokens(cur_s_target)

            align_t_idx, align_s_idx = align_sequences(
                cur_t_target_tokens, 
                cur_s_target_tokens, 
                distiller.student_tokenizer, 
                distiller.teacher_tokenizer
            )
            if align_t_idx == [] and align_s_idx == []:
                t_preds_as_label.append(torch.tensor([-100] * t_preds.shape[-1], device=tea_v_hiddens.device))
                align_ratio.append(1.0)
                continue
            cur_align_ratio = len(align_s_idx) / len(cur_s_target)
            align_ratio.append(cur_align_ratio)

            cur_t_preds_as_label_1 = target[i][:indices_s_target[0]].cpu().tolist()
            cur_t_preds_as_label_2 = [-100] * len(cur_s_target)
            for _t_idx, _s_idx in zip(align_t_idx, align_s_idx):
                tmp_t_token = distiller.teacher_tokenizer.convert_ids_to_tokens([cur_t_preds[_t_idx]])
                if cur_t_preds[_t_idx] == distiller.teacher_tokenizer.eos_token_id:
                    cur_t_preds_as_label_2[_s_idx] = distiller.student_tokenizer.eos_token_id
                    s2t_hiddens_align[i, _t_idx + indices_t_target[0], :] = stu_v_hiddens[i, _s_idx + indices_s_target[0]]
                    t2s_hiddens_align[i, _s_idx + indices_s_target[0], :] = tea_v_hiddens[i, _t_idx + indices_t_target[0]]
                else:
                    try:
                        tmp = distiller.student_tokenizer.convert_tokens_to_ids(tmp_t_token)
                        if len(tmp) == 1 and tmp[0] is not None:
                            cur_t_preds_as_label_2[_s_idx] = tmp[0]
                            s2t_hiddens_align[i, _t_idx + indices_t_target[0], :] = stu_v_hiddens[i, _s_idx + indices_s_target[0]]
                            t2s_hiddens_align[i, _s_idx + indices_s_target[0], :] = tea_v_hiddens[i, _t_idx + indices_t_target[0]]
                        else:
                            cur_t_preds_as_label_2[_s_idx] = -100
                    except:
                        cur_t_preds_as_label_2[_s_idx] = -100

            assert len(cur_t_preds_as_label_2) == len(cur_s_target)
            cur_t_preds_as_label_3 = target[i][indices_s_target[-1]+1:].cpu().tolist()
            
            cur_t_preds_as_label = cur_t_preds_as_label_1 + cur_t_preds_as_label_2 + cur_t_preds_as_label_3
            cur_t_preds_as_label = cur_t_preds_as_label[:teacher_target.shape[1]]
            t_preds_as_label.append(torch.tensor(cur_t_preds_as_label, device=teacher_target.device))

        t_preds_as_label = torch.cat(t_preds_as_label, dim=0).reshape(-1, teacher_target.shape[1])
        log["align_ratio"] = torch.tensor(align_ratio, device=teacher_target.device).mean()
        t2s_logits = t2s_hiddens_align.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )  # n x d x d x V_stu -> n x V_stu  [bsz x seq-len x V_stu]

        stu_align_token_num = max(1e-3, t_preds_as_label.ne(-100).sum())
        
        t2s_agreement_mask = t2s_logits.argmax(-1).eq(t_preds_as_label)
        stu_aggreement_num = max(1e-3, t2s_agreement_mask.sum())

        t2s_agreement = (t2s_agreement_mask * t_preds_as_label.ne(-100)).sum() / stu_align_token_num
        t2s_agreement_ratio = t2s_agreement_mask.sum() / pad_mask.sum()
        log["t2s_agreement"] = t2s_agreement
        log["t2s_agreement_ratio"] = t2s_agreement_ratio

        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum() / pad_mask.sum()
        t2s_acc_ratio = t2s_acc_mask.sum() / pad_mask.sum()
        log["t2s_acc"] = t2s_acc
        log["t2s_acc_ratio"] = t2s_acc_ratio
        log["t_preds_as_label_acc"] = (t_preds_as_label.eq(target) * t_preds_as_label.ne(-100)).sum() / stu_align_token_num

        t2s_ce_loss = self.compute_cross_entropy_loss(
            t2s_logits, t_preds_as_label, reduction="sum"
        )[0] / stu_align_token_num
        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none"
        )
        if t2s_agreement <= distiller.args.t2s_agreement:
            t2s_kd_loss = (t2s_kd_loss * t2s_agreement_mask).sum() / stu_aggreement_num
        else:
            t2s_kd_loss = (t2s_kd_loss * t_preds_as_label.ne(-100)).sum() / stu_align_token_num
        
        log["t2s_ce_loss"] = t2s_ce_loss
            
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens_align)
        s2t_kd_loss = self.dist_func(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * (~s2t_hiddens_align.eq(0).all(-1))).sum() / max((~s2t_hiddens_align.eq(0).all(-1)).sum(), 1e-8)

        if self.args.only_stu_kd:
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif self.args.only_tea_kd:
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss
        
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["kd_loss"] = kd_loss

        return kd_loss, log
   

    def compute_on_policy_dual_space_kd_loss_with_eta(
        self, outputs, teacher_outputs, batch, distiller, log
    ):
        target = batch["op_label_batch"]["label"]
        teacher_target = batch["op_teacher_label_batch"]["label"]
          
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]
        
        # teacher space
        if self.args.init_s2t_projector:
            stu_lm_head = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
            if self.args.topk_vocab != -1:
                stu_lm_head = stu_lm_head[:, :self.args.topk_vocab]
            s2t_projector = stu_lm_head @ distiller.part_teacher_head_pinv
            s2t_hiddens = hiddens @ s2t_projector
        else:
            stu_v_hiddens = distiller.s2t_projector(hiddens)

        tea_v_hiddens = distiller.t2s_projector(teacher_hiddens)  # m x D x D x d -> m x d

        t_preds = teacher_outputs.logits.argmax(-1)
        t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)

        t_preds_as_label = []
        t2s_hiddens_align = torch.zeros_like(tea_v_hiddens).to(tea_v_hiddens.device).to(tea_v_hiddens)   # m x d -> n x d
        s2t_hiddens_align = torch.zeros_like(stu_v_hiddens).to(stu_v_hiddens.device).to(stu_v_hiddens)   # n x D -> m x D
        align_ratio = []
        for i in range(t_preds.shape[0]):
            indices_t_preds = torch.where(t_preds[i] != -100)[0]
            if indices_t_preds.shape[0] == 0:
                t_preds_as_label.append(torch.tensor([-100]*t_preds.shape[-1], device=tea_v_hiddens.device))
                align_ratio.append(1.0)
                continue
            indices_t_target = torch.where(teacher_target[i] != -100)[0]
            indices_s_target = torch.where(target[i] != -100)[0]

            cur_t_preds = t_preds[i][indices_t_preds[0]: indices_t_preds[-1]+1]
            cur_t_target = teacher_target[i][indices_t_target[0]: indices_t_target[-1]+1]
            cur_t_target_tokens = distiller.teacher_tokenizer.convert_ids_to_tokens(cur_t_target)

            cur_s_target = target[i][indices_s_target[0]: indices_s_target[-1]+1]
            cur_s_target_tokens = distiller.student_tokenizer.convert_ids_to_tokens(cur_s_target)

            align_t_idx, align_s_idx = align_sequences(
                cur_t_target_tokens, 
                cur_s_target_tokens,
                distiller.student_tokenizer, 
                distiller.teacher_tokenizer
            )
            if align_t_idx == [] and align_s_idx == []:
                t_preds_as_label.append(torch.tensor([-100]*t_preds.shape[-1], device=tea_v_hiddens.device))
                align_ratio.append(1.0)
                continue
            cur_align_ratio = len(align_s_idx) / len(cur_s_target)
            align_ratio.append(cur_align_ratio)

            cur_t_preds_as_label_1 = target[i][:indices_s_target[0]].cpu().tolist()
            cur_t_preds_as_label_2 = [-100] * len(cur_s_target)
            for _t_idx, _s_idx in zip(align_t_idx, align_s_idx):
                tmp_t_token = distiller.teacher_tokenizer.convert_ids_to_tokens([cur_t_preds[_t_idx]])
                if cur_t_preds[_t_idx] == distiller.teacher_tokenizer.eos_token_id:
                    cur_t_preds_as_label_2[_s_idx] = distiller.student_tokenizer.eos_token_id
                    s2t_hiddens_align[i, _t_idx + indices_t_target[0], :] = stu_v_hiddens[i, _s_idx + indices_s_target[0]]
                    t2s_hiddens_align[i, _s_idx + indices_s_target[0], :] = tea_v_hiddens[i, _t_idx + indices_t_target[0]]
                else:
                    try:
                        tmp = distiller.student_tokenizer.convert_tokens_to_ids(tmp_t_token)
                        if len(tmp) == 1 and tmp[0] is not None:
                            cur_t_preds_as_label_2[_s_idx] = tmp[0]
                            s2t_hiddens_align[i, _t_idx + indices_t_target[0], :] = stu_v_hiddens[i, _s_idx + indices_s_target[0]]
                            t2s_hiddens_align[i, _s_idx + indices_s_target[0], :] = tea_v_hiddens[i, _t_idx + indices_t_target[0]]
                        else:
                            cur_t_preds_as_label_2[_s_idx] = -100
                    except:
                        cur_t_preds_as_label_2[_s_idx] = -100

            assert len(cur_t_preds_as_label_2) == len(cur_s_target)
            cur_t_preds_as_label_3 = target[i][indices_s_target[-1]+1:].cpu().tolist()
            
            cur_t_preds_as_label = cur_t_preds_as_label_1 + cur_t_preds_as_label_2 + cur_t_preds_as_label_3
            cur_t_preds_as_label = cur_t_preds_as_label[:teacher_target.shape[1]]
            t_preds_as_label.append(torch.tensor(cur_t_preds_as_label, device=teacher_target.device))

        t_preds_as_label = torch.cat(t_preds_as_label, dim=0).reshape(-1, teacher_target.shape[1])
        log["align_ratio"] = torch.tensor(align_ratio, device=teacher_target.device).mean()

        t2s_logits = t2s_hiddens_align.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )  # n x d x d x V_stu -> n x V_stu  [bsz x seq-len x V_stu]

        stu_align_token_num = max(1e-3, t_preds_as_label.ne(-100).sum())
        t2s_agreement_mask = t2s_logits.argmax(-1).eq(t_preds_as_label)
        stu_aggreement_num = max(1e-3, t2s_agreement_mask.sum())

        t2s_agreement = (t2s_agreement_mask * t_preds_as_label.ne(-100)).sum() / stu_align_token_num
        t2s_agreement_ratio = t2s_agreement_mask.sum() / pad_mask.sum()
        log["t2s_agreement"] = t2s_agreement
        log["t2s_agreement_ratio"] = t2s_agreement_ratio

        t2s_ce_loss = self.compute_cross_entropy_loss(
            t2s_logits, t_preds_as_label, reduction="sum"
        )[0] / stu_align_token_num
        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none"
        )
        if t2s_agreement <= distiller.args.t2s_agreement:
            t2s_kd_loss = (t2s_kd_loss * t2s_agreement_mask).sum() / stu_aggreement_num
        else:
            t2s_kd_loss = (t2s_kd_loss * t_preds_as_label.ne(-100)).sum() / stu_align_token_num

        log["t2s_ce_loss"] = t2s_ce_loss
            
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens_align)
        s2t_kd_loss = self.dist_func(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * (~s2t_hiddens_align.eq(0).all(-1))).sum() / max((~s2t_hiddens_align.eq(0).all(-1)).sum(), 1e-8)

        if self.args.only_stu_kd:
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif self.args.only_tea_kd:
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss
        
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["kd_loss"] = kd_loss

        return kd_loss, log
   
