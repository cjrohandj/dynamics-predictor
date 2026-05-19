"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def _training_step(model) -> int:
    step = int(getattr(model, "_student_loss_step", 0)) + 1
    setattr(model, "_student_loss_step", step)
    return step


def _linear_ramp(step: int, start_after: int, ramp_updates: int) -> float:
    if ramp_updates <= 0:
        return 1.0
    return max(0.0, min(1.0, (step - start_after) / ramp_updates))


def _scheduled_float(
    step: int,
    start_value: float,
    end_value: float,
    start_after: int,
    ramp_updates: int,
    pre_start_value: float | None = None,
) -> float:
    if pre_start_value is not None and step < start_after:
        return pre_start_value
    progress = _linear_ramp(step, start_after, ramp_updates)
    return start_value + progress * (end_value - start_value)


def _scheduled_loss_settings(model, cfg: dict) -> tuple[float, float, int, float]:
    loss_cfg = cfg["loss"]
    step = _training_step(model)
    curriculum = loss_cfg.get("curriculum", {})
    if not bool(curriculum.get("enabled", False)):
        return (
            float(loss_cfg.get("one_step_weight", 1.0)),
            float(loss_cfg.get("rollout_weight", 0.3)),
            int(loss_cfg.get("rollout_train_horizon", 5)),
            1.0,
        )

    progress = _linear_ramp(
        step,
        int(curriculum.get("start_after_updates", 0)),
        int(curriculum.get("ramp_updates", 1)),
    )
    one_start = float(curriculum.get("one_step_weight_start", loss_cfg.get("one_step_weight", 1.0)))
    one_end = float(curriculum.get("one_step_weight_end", loss_cfg.get("one_step_weight", one_start)))
    roll_start = float(curriculum.get("rollout_weight_start", loss_cfg.get("rollout_weight", 0.3)))
    roll_end = float(curriculum.get("rollout_weight_end", loss_cfg.get("rollout_weight", roll_start)))
    horizon_start = int(curriculum.get("rollout_horizon_start", loss_cfg.get("rollout_train_horizon", 5)))
    horizon_end = int(curriculum.get("rollout_horizon_end", loss_cfg.get("rollout_train_horizon", horizon_start)))
    one_weight = one_start + progress * (one_end - one_start)
    if "rollout_weight_start_after_updates" in curriculum or "rollout_weight_ramp_updates" in curriculum:
        rollout_weight = _scheduled_float(
            step,
            roll_start,
            roll_end,
            int(curriculum.get("rollout_weight_start_after_updates", curriculum.get("start_after_updates", 0))),
            int(curriculum.get("rollout_weight_ramp_updates", curriculum.get("ramp_updates", 1))),
            float(curriculum["rollout_weight_pre_start"]) if "rollout_weight_pre_start" in curriculum else None,
        )
    else:
        rollout_weight = roll_start + progress * (roll_end - roll_start)
    if "rollout_horizon_start_after_updates" in curriculum or "rollout_horizon_ramp_updates" in curriculum:
        horizon = int(
            round(
                _scheduled_float(
                    step,
                    float(horizon_start),
                    float(horizon_end),
                    int(curriculum.get("rollout_horizon_start_after_updates", curriculum.get("start_after_updates", 0))),
                    int(curriculum.get("rollout_horizon_ramp_updates", curriculum.get("ramp_updates", 1))),
                )
            )
        )
    else:
        horizon = int(round(horizon_start + progress * (horizon_end - horizon_start)))
    return one_weight, rollout_weight, max(1, horizon), progress


def _scheduled_vpt_weight(loss_cfg: dict, step: int) -> tuple[float, float]:
    base_weight = float(loss_cfg.get("vpt_surrogate_weight", 0.0))
    schedule = loss_cfg.get("vpt_surrogate_schedule", {})
    if not bool(schedule.get("enabled", False)):
        return base_weight, 1.0

    progress = _linear_ramp(
        step,
        int(schedule.get("start_after_updates", 0)),
        int(schedule.get("ramp_updates", 1)),
    )
    start_weight = float(schedule.get("weight_start", 0.0))
    end_weight = float(schedule.get("weight_end", base_weight))
    weight = start_weight + progress * (end_weight - start_weight)
    return weight, progress


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    loss_type: str = "mse",
    huber_beta: float = 1.0,
) -> torch.Tensor:
    # Train local open-loop stability at random positions, not only at the
    # beginning of each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    if loss_type == "huber":
        return F.smooth_l1_loss(pred_norm, target_norm, beta=float(huber_beta))
    return F.mse_loss(pred_norm, target_norm)


def vpt_surrogate_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    loss_cfg: dict,
) -> torch.Tensor:
    """Differentiable proxy for VPT80@0.25.

    Official VPT80@0.25 asks whether at least 80% of windows stay below
    step-wise nMSE 0.25. The hard threshold and percentile are not useful as a
    direct training loss, so this penalizes the 80th percentile step nMSE when
    it approaches or crosses the threshold.
    """
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for VPT surrogate loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    obs_std = torch.as_tensor(normalizer.obs_std, dtype=preds.dtype, device=preds.device).clamp_min(1e-6)
    step_nmse = torch.mean(((preds - targets) / obs_std) ** 2, dim=-1)
    percentile = float(loss_cfg.get("vpt_percentile", 0.80))
    threshold = float(loss_cfg.get("vpt_threshold", 0.25))
    margin = max(float(loss_cfg.get("vpt_margin", 0.05)), 1e-6)
    q_nmse = torch.quantile(step_nmse, percentile, dim=0)
    return F.softplus((q_nmse - threshold) / margin).mean() * margin


def _multi_rollout_horizons(loss_cfg: dict, horizon: int) -> list[int]:
    raw = loss_cfg.get("multi_rollout_horizons")
    if raw is None:
        return [int(horizon)]
    horizons = [int(h) for h in raw]
    if int(horizon) not in horizons:
        horizons.append(int(horizon))
    return sorted({max(1, h) for h in horizons if h <= int(horizon)})


def multi_rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    loss_cfg: dict,
) -> tuple[torch.Tensor, list[int]]:
    horizons = _multi_rollout_horizons(loss_cfg, horizon)
    losses = [
        rollout_loss(
            model,
            states,
            actions,
            normalizer,
            warmup_steps=warmup_steps,
            horizon=h,
            loss_type=str(loss_cfg.get("rollout_loss_type", "mse")),
            huber_beta=float(loss_cfg.get("huber_beta", 1.0)),
        )
        for h in horizons
    ]
    return torch.stack(losses).mean(), horizons


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    one = one_step_delta_loss(model, states, actions, normalizer)
    one_weight, rollout_weight, horizon, progress = _scheduled_loss_settings(model, cfg)
    step = int(getattr(model, "_student_loss_step", 0))
    warmup = int(cfg["eval"].get("warmup_steps", 5))
    roll, rollout_horizons = multi_rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=horizon,
        loss_cfg=loss_cfg,
    )
    vpt_weight, vpt_progress = _scheduled_vpt_weight(loss_cfg, step)
    if vpt_weight > 0.0:
        vpt = vpt_surrogate_loss(
            model,
            states,
            actions,
            normalizer,
            warmup_steps=warmup,
            horizon=int(loss_cfg.get("vpt_surrogate_horizon", horizon)),
            loss_cfg=loss_cfg,
        )
    else:
        vpt = states.new_tensor(0.0)
    total = one_weight * one + rollout_weight * roll + vpt_weight * vpt
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/vpt_surrogate": float(vpt.detach().cpu()),
        "loss/vpt_surrogate_weight": vpt_weight,
        "loss/vpt_surrogate_progress": vpt_progress,
        "loss/one_step_weight": one_weight,
        "loss/rollout_weight": rollout_weight,
        "loss/rollout_horizon": float(horizon),
        "loss/rollout_num_horizons": float(len(rollout_horizons)),
        "loss/curriculum_progress": progress,
    }
