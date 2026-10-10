import torch

from verl.trainer.ppo.core_algos import compute_self_distillation_loss


def build_apsd_target(student_log_probs, teacher_log_probs, correct_mask, alpha_correct, alpha_incorrect):
    if student_log_probs.ndim != 3 or student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("Student and teacher must have matching [batch, response, vocabulary] shapes")
    if correct_mask.shape != student_log_probs.shape[:1] or correct_mask.dtype != torch.bool:
        raise ValueError("correct_mask must be boolean with shape [batch]")
    dtype = torch.promote_types(student_log_probs.dtype, torch.float32)
    log_p = student_log_probs.to(dtype)
    log_q = teacher_log_probs.detach().to(dtype)
    alpha = torch.where(
        correct_mask.to(log_p.device),
        torch.as_tensor(alpha_correct, dtype=dtype, device=log_p.device),
        torch.as_tensor(alpha_incorrect, dtype=dtype, device=log_p.device),
    )
    alpha = alpha[:, None, None]
    unnormalized = (1 - alpha) * log_p + alpha * log_q
    return unnormalized - torch.logsumexp(unnormalized, dim=-1, keepdim=True)


def compute_apsd_loss(
    student_log_probs,
    teacher_log_probs,
    response_mask,
    self_distillation_config,
    apsd_config,
    correct_mask,
    student_all_log_probs,
    teacher_all_log_probs,
    old_log_probs=None,
    self_distillation_mask=None,
    loss_agg_mode="token-mean",
    rollout_is_weights=None,
):
    if not self_distillation_config.full_logit_distillation:
        raise ValueError("APSD requires full_logit_distillation=True to construct its target distribution")
    if student_all_log_probs is None or teacher_all_log_probs is None:
        raise ValueError("APSD needs full-vocabulary probabilities before top-k projection")
    log_target = build_apsd_target(
        student_all_log_probs, teacher_all_log_probs, correct_mask,
        apsd_config.alpha_correct, apsd_config.alpha_incorrect,
    )
    log_p = student_all_log_probs.to(log_target.dtype)
    student_topk = target_topk = None
    k = self_distillation_config.distillation_topk
    if k is not None:
        indices = torch.topk(log_p, min(k, log_p.shape[-1]), dim=-1).indices
        student_topk = log_p.gather(-1, indices)
        target_topk = log_target.gather(-1, indices)
    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs.detach(),
        response_mask=response_mask,
        self_distillation_config=self_distillation_config,
        old_log_probs=old_log_probs,
        student_all_log_probs=log_p,
        teacher_all_log_probs=log_target,
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=target_topk,
        self_distillation_mask=self_distillation_mask,
        loss_agg_mode=loss_agg_mode,
        rollout_is_weights=rollout_is_weights,
    )
    metrics["apsd/correct_rollout_fraction"] = correct_mask.float().mean().item()
    metrics["apsd/alpha_correct"] = apsd_config.alpha_correct
    metrics["apsd/alpha_incorrect"] = apsd_config.alpha_incorrect
    return loss, metrics
