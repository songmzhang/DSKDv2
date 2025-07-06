import torch
import torch.nn.functional as F
from .various_divergence import VariousDivergence
from utils import print_rank


class DualSpaceKDV2(VariousDivergence):
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
            kd_loss, log = self.compute_on_policy_dual_space_kd_loss(
                outputs, teacher_outputs, batch, distiller, log
            )
        else:    # off-policy scenario
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **batch["teacher_input_batch"], 
                    output_hidden_states=True
                )
            kd_loss, log = self.compute_dual_space_kd_loss(
                outputs, teacher_outputs, batch, distiller, log
            )
        loss = (1.0 - self.kd_rate) * ce_loss + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(logits, batch["label_batch"])
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, log)
        return loss, logging_output

    def compute_dual_space_kd_loss(
        self, outputs, teacher_outputs, batch, distiller, log
    ):
        target = batch["label_batch"]["label"]
        pad_mask = target.ne(self.padding_id)
        teacher_target = batch["teacher_label_batch"]["label"]
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        # student space
        t2s_hiddens = distiller.t2s_projector(teacher_hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )
        
        t_preds = teacher_outputs.logits.argmax(-1)
        t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)
        t2s_ce_loss = self.compute_cross_entropy_loss(
            t2s_logits, t_preds, reduction="sum"
        )[0] / batch["label_batch"]["loss_denom"]
        
        # To figure out how many teacher distributions are consistent before&after projection
        t2s_agreement_mask = t2s_logits.argmax(-1).eq(t_preds) * pad_mask
        t2s_agreement = t2s_agreement_mask.sum() / batch["label_batch"]["loss_denom"]

        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none"
        )
        
        # mask the projected distributions that are not consistent with itself before projection
        if t2s_agreement <= distiller.args.t2s_agreement:
            t2s_kd_loss = (t2s_kd_loss * t2s_agreement_mask).sum() / max(1e-3, t2s_agreement_mask.sum())
        else:
            t2s_kd_loss = (t2s_kd_loss * pad_mask).sum() / batch["label_batch"]["loss_denom"]

        # teacher space
        if self.args.init_s2t_projector:
            stu_lm_head = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
            if self.args.topk_vocab != -1:
                stu_lm_head = stu_lm_head[:, :self.args.topk_vocab]
            distiller.s2t_projector = stu_lm_head @ distiller.part_teacher_head_pinv
            s2t_hiddens = hiddens @ distiller.s2t_projector
        else:
            s2t_hiddens = distiller.s2t_projector(hiddens)
            
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)
        
        s2t_kd_loss = self.dist_func(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum() / batch["label_batch"]["loss_denom"]
        
        if self.args.only_stu_kd:
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif self.args.only_tea_kd:
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss
        
        s2t_agreement = (s2t_logits.argmax(-1).eq(outputs.logits.argmax(-1)) * pad_mask).sum() / batch["label_batch"]["loss_denom"]
        t_acc = (teacher_outputs.logits.argmax(-1).eq(target) * pad_mask).sum() / batch["label_batch"]["loss_denom"]
        t2s_acc = (t2s_logits.argmax(-1).eq(target) * pad_mask).sum() / batch["label_batch"]["loss_denom"]

        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["t2s_agreement"] = t2s_agreement
        log["s2t_agreement"] = s2t_agreement
        log["t_acc"] = t_acc
        log["t2s_acc"] = t2s_acc
        log["kd_loss"] = kd_loss
        return kd_loss, log
    
    def compute_on_policy_dual_space_kd_loss(
        self, 
        outputs, 
        teacher_outputs, 
        batch,
        distiller, 
        log
    ):
        target = batch["op_label_batch"]["label"]
        pad_mask = target.ne(self.padding_id)
        teacher_target = batch["op_teacher_label_batch"]["label"]
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        # student space
        # use model-genereated text for projector training
        t2s_hiddens = distiller.t2s_projector(teacher_hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )
        
        # We use t_preds as labels here to make sure that the projected teacher distribution 
        # is consistent with original teacher distribution
        t_preds = teacher_outputs.logits.argmax(-1)
        t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)
        t2s_ce_loss = self.compute_cross_entropy_loss(
            t2s_logits, t_preds, reduction="sum"
        )[0] / batch["op_label_batch"]["loss_denom"]
        
        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), batch["op_label_batch"]["label"], reduction="none"
        )
        t2s_kd_loss = (t2s_kd_loss * pad_mask).sum() / batch["op_label_batch"]["loss_denom"]

        # teacher space
        # calculate s2t_projector for each step with student LM head and teacher LM head
        if self.args.init_s2t_projector:
            stu_lm_head = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
            if self.args.topk_vocab != -1:
                stu_lm_head = stu_lm_head[:, :self.args.topk_vocab]
            distiller.s2t_projector = stu_lm_head @ distiller.part_teacher_head_pinv
            s2t_hiddens = hiddens @ distiller.s2t_projector
        else:
            s2t_hiddens = distiller.s2t_projector(hiddens)
        
        s2t_hiddens = hiddens @ distiller.s2t_projector
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)
        
        s2t_kd_loss = self.dist_func(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum() / batch["op_label_batch"]["loss_denom"]
        
        if self.args.only_stu_kd:
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif self.args.only_tea_kd:
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss
        
        t2s_agreement = (t2s_logits.argmax(-1).eq(t_preds) * pad_mask).sum() / batch["op_label_batch"]["loss_denom"]
        s2t_agreement = (s2t_logits.argmax(-1).eq(outputs.logits.argmax(-1)) * pad_mask).sum() / batch["label_batch"]["loss_denom"]
        
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["t2s_agreement"] = t2s_agreement
        log["s2t_agreement"] = t2s_agreement
        log["kd_loss"] = kd_loss
        return kd_loss, log
    
    