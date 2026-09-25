"""Small, dependency-light G2D-UOT primitives used by ThyQC training."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch


def flatten_task_probs(probs: Mapping[str, torch.Tensor], task_order: Sequence[str]) -> torch.Tensor:
    """Concatenate per-task categorical probabilities in a deterministic order."""
    return torch.cat([probs[name] for name in task_order], dim=-1)


def _generalized_kl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Generalized KL for non-negative, not necessarily normalized masses."""
    x = x.clamp_min(1e-12)
    y = y.clamp_min(1e-12)
    return (x * (x.log() - y.log()) - x + y).sum(dim=-1)


def unbalanced_sinkhorn_objective(
    source: torch.Tensor,
    target: torch.Tensor,
    cost: torch.Tensor,
    *,
    epsilon: float = 0.08,
    rho: float = 0.50,
    iterations: int = 30,
    return_components: bool = False,
):
    """Full differentiable entropic UOT objective.

    ``source`` and ``target`` are non-negative ``[B,K]`` masses and ``cost`` is
    ``[K,K]``.  The source is detached because it is an offline teacher mass;
    gradients flow only through the student target.
    """
    if source.ndim != 2 or target.ndim != 2 or cost.ndim not in (2, 3):
        raise ValueError("source/target must be [B,K] and cost [K,K] or [B,K,K]")
    expected = (source.shape[-1], source.shape[-1])
    if source.shape != target.shape or cost.shape[-2:] != expected:
        raise ValueError("incompatible source, target, and cost shapes")
    if cost.ndim == 3 and cost.shape[0] != source.shape[0]:
        raise ValueError("batched cost must have the same batch size as masses")
    eps = max(float(epsilon), 1e-5)
    tau = float(rho) / (float(rho) + eps)
    a = source.detach().clamp_min(1e-8)
    b = target.clamp_min(1e-8)
    a = a / a.sum(dim=-1, keepdim=True)
    b = b / b.sum(dim=-1, keepdim=True)
    batch_cost = cost.detach().to(target).clamp_min(0.0)
    if batch_cost.ndim == 2:
        batch_cost = batch_cost.unsqueeze(0).expand(source.shape[0], -1, -1)
    kernel = torch.exp((-batch_cost / eps).clamp(min=-30.0, max=30.0)).clamp_min(1e-12)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(int(iterations)):
        kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1)
        u = (a / kv.clamp_min(1e-8)).pow(tau).clamp(1e-8, 1e8)
        ktu = torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
        v = (b / ktu.clamp_min(1e-8)).pow(tau).clamp(1e-8, 1e8)
    plan = u.unsqueeze(-1) * kernel * v.unsqueeze(1)
    row_mass = plan.sum(dim=-1)
    col_mass = plan.sum(dim=-2)
    transport = (plan * batch_cost).sum(dim=(-1, -2)).mean()
    source_kl = float(rho) * _generalized_kl(row_mass, a).mean()
    target_kl = float(rho) * _generalized_kl(col_mass, b).mean()
    # H(pi) = -sum pi(log pi - 1), hence -epsilon*H is below.
    negative_entropy = eps * (plan * (plan.clamp_min(1e-12).log() - 1.0)).sum(dim=(-1, -2)).mean()
    parts = {"transport": transport, "source_kl": source_kl,
             "target_kl": target_kl, "negative_entropy": negative_entropy}
    objective = sum(parts.values())
    if return_components:
        return objective, {name: value.detach() for name, value in parts.items()}
    return objective


def unbalanced_sinkhorn_cost(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible alias for the full UOT objective."""
    return unbalanced_sinkhorn_objective(*args, **kwargs)


def unbalanced_sinkhorn_transport_discrepancy(
    source: torch.Tensor,
    target: torch.Tensor,
    cost: torch.Tensor,
    *,
    epsilon: float = 0.08,
    rho: float = 0.50,
    iterations: int = 30,
) -> torch.Tensor:
    """Transport discrepancy induced by a relaxed-marginal Sinkhorn plan.

    The plan is still unbalanced: ``rho`` controls how strongly its marginals
    follow teacher and student masses.  Only the clinically meaningful
    transport work is exposed to the outer training objective; KL relaxation
    and entropy are solver regularizers rather than extra rewards/penalties.
    """
    if source.ndim != 2 or target.ndim != 2 or cost.ndim not in (2, 3):
        raise ValueError("source/target must be [B,K] and cost [K,K] or [B,K,K]")
    if source.shape != target.shape or cost.shape[-2:] != (source.shape[-1], source.shape[-1]):
        raise ValueError("incompatible source, target, and cost shapes")
    if cost.ndim == 3 and cost.shape[0] != source.shape[0]:
        raise ValueError("batched cost must have the same batch size as masses")
    eps = max(float(epsilon), 1e-5)
    tau = float(rho) / (float(rho) + eps)
    a = source.detach().clamp_min(1e-8)
    b = target.clamp_min(1e-8)
    a = a / a.sum(dim=-1, keepdim=True)
    b = b / b.sum(dim=-1, keepdim=True)
    batch_cost = cost.detach().to(target).clamp_min(0.0)
    if batch_cost.ndim == 2:
        batch_cost = batch_cost.unsqueeze(0).expand(source.shape[0], -1, -1)
    kernel = torch.exp((-batch_cost / eps).clamp(min=-30.0, max=30.0)).clamp_min(1e-12)
    u, v = torch.ones_like(a), torch.ones_like(b)
    for _ in range(int(iterations)):
        u = (a / torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)).pow(tau).clamp(1e-8, 1e8)
        v = (b / torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)).pow(tau).clamp(1e-8, 1e8)
    plan = u.unsqueeze(-1) * kernel * v.unsqueeze(1)
    return (plan * batch_cost).sum(dim=(-1, -2)).mean()


def contextual_semantic_cost(
    prototypes: torch.Tensor,
    contexts: torch.Tensor,
    *,
    context_weight: float = 0.5,
) -> torch.Tensor:
    """Return per-sample cosine costs between rationale atoms and prototypes.

    ``prototypes`` is ``[K,D]``. ``contexts`` may be one rationale embedding per
    sample (``[B,D]``) or one per source atom (``[B,K,D]``).
    """
    if prototypes.ndim != 2 or contexts.ndim not in (2, 3):
        raise ValueError("prototypes [K,D], contexts [B,D] or [B,K,D] required")
    q = torch.nn.functional.normalize(prototypes.float(), dim=-1)
    if contexts.ndim == 2:
        contexts = contexts.unsqueeze(1).expand(-1, q.shape[0], -1)
    if contexts.shape[1:] != q.shape:
        raise ValueError("context atom count/embedding dimension mismatch")
    source_atoms = torch.nn.functional.normalize(
        q.unsqueeze(0) + float(context_weight) * contexts.float(), dim=-1
    )
    return (1.0 - torch.einsum("bjd,kd->bjk", source_atoms, q)).clamp(0.0, 2.0)


def probability_conditioned_semantic_cost(
    prototypes: torch.Tensor,
    teacher_mass: torch.Tensor,
    *,
    context_weight: float = 2.0,
) -> torch.Tensor:
    """Build sample-specific semantic geometry from the full teacher belief.

    Each class prototype is a short structured-rationale embedding.  Their
    teacher-probability-weighted barycentre therefore retains uncertainty that
    an argmax rationale string discards, while remaining a semantic cost rather
    than an additional probability loss.
    """
    if prototypes.ndim != 2 or teacher_mass.ndim != 2:
        raise ValueError("prototypes [K,D] and teacher_mass [B,K] required")
    if teacher_mass.shape[-1] != prototypes.shape[0]:
        raise ValueError("teacher mass and prototype count mismatch")
    q = torch.nn.functional.normalize(prototypes.float(), dim=-1)
    mass = teacher_mass.float().clamp_min(0.0)
    mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    semantic_context = mass @ q
    return contextual_semantic_cost(
        q, semantic_context, context_weight=context_weight
    )


RATIONALE_TOKENS = {
    "acceptable": {"clear", "diagnostic", "usable", "complete"},
    "limited": {"partly", "usable", "incomplete", "adjust"},
    "unacceptable": {"non-diagnostic", "artifact", "repeat"},
    "adequate": {"thyroid", "visible", "complete"},
    "partial": {"thyroid", "visible", "incomplete", "adjust"},
    "absent": {"thyroid", "not-visible", "reacquire"},
    "uncertain": {"thyroid", "uncertain", "adjust"},
    "yes": {"keyframe", "diagnostic", "complete", "hold"},
    "candidate": {"keyframe", "incomplete", "adjust"},
    "no": {"not-keyframe", "repeat", "reacquire"},
    "none": {"no-defect", "diagnostic", "hold"},
    "severe_artifact": {"artifact", "non-diagnostic", "repeat"},
    "thyroid_not_visible": {"thyroid", "not-visible", "reacquire"},
    "target_incomplete": {"thyroid", "incomplete", "adjust"},
    "hold": {"diagnostic", "complete", "hold"},
    "repeat_acquisition": {"artifact", "repeat"},
    "reacquire_target": {"thyroid", "not-visible", "reacquire"},
    "adjust_angle_and_rescan": {"incomplete", "adjust"},
}


def semantic_cost_matrix(tasks: Mapping[str, Sequence[str]]) -> torch.Tensor:
    """Cosine-like Jaccard distance between short structured rationales."""
    names = [(task, label) for task, labels in tasks.items() for label in labels]
    n = len(names)
    c = torch.zeros((n, n), dtype=torch.float32)
    for i, (task_i, label_i) in enumerate(names):
        for j, (task_j, label_j) in enumerate(names):
            a, b = RATIONALE_TOKENS[label_i], RATIONALE_TOKENS[label_j]
            c[i, j] = 1.0 - len(a & b) / max(1, len(a | b))
            if task_i == task_j and label_i == label_j:
                c[i, j] = 0.0
    return c


def clinical_relation_cost_matrix(tasks: Mapping[str, Sequence[str]]) -> torch.Tensor:
    """Encode legacy decision-chain and KG rules only as C^clin."""
    names = [(task, label) for task, labels in tasks.items() for label in labels]
    n = len(names)
    c = torch.full((n, n), 0.50, dtype=torch.float32)
    c.fill_diagonal_(0.0)
    mappings = {
        ("primary_defect", "none"): [("image_usability", "acceptable"), ("keyframe_status", "yes"), ("recommended_action", "hold")],
        ("primary_defect", "severe_artifact"): [("image_usability", "unacceptable"), ("keyframe_status", "no"), ("recommended_action", "repeat_acquisition")],
        ("primary_defect", "thyroid_not_visible"): [("image_usability", "acceptable"), ("keyframe_status", "no"), ("recommended_action", "reacquire_target")],
        ("primary_defect", "target_incomplete"): [("image_usability", "limited"), ("keyframe_status", "candidate"), ("recommended_action", "adjust_angle_and_rescan")],
        ("structure_visibility", "adequate"): [("primary_defect", "none"), ("recommended_action", "hold")],
        ("structure_visibility", "partial"): [("primary_defect", "target_incomplete"), ("recommended_action", "adjust_angle_and_rescan")],
        ("structure_visibility", "absent"): [("primary_defect", "thyroid_not_visible"), ("recommended_action", "reacquire_target")],
        ("structure_visibility", "uncertain"): [("primary_defect", "target_incomplete"), ("recommended_action", "adjust_angle_and_rescan")],
        ("image_usability", "acceptable"): [("recommended_action", "hold")],
        ("image_usability", "limited"): [("recommended_action", "adjust_angle_and_rescan")],
        ("image_usability", "unacceptable"): [("recommended_action", "repeat_acquisition")],
        ("keyframe_status", "yes"): [("recommended_action", "hold")],
        ("keyframe_status", "candidate"): [("recommended_action", "adjust_angle_and_rescan")],
        ("keyframe_status", "no"): [("recommended_action", "repeat_acquisition")],
    }
    index = {x: i for i, x in enumerate(names)}
    for src, dsts in mappings.items():
        for dst in dsts:
            if src in index and dst in index:
                c[index[src], index[dst]] = 0.0
                c[index[dst], index[src]] = 0.0
    # Known incompatibilities receive a high transport cost rather than a
    # separate penalty.  This is the KG/decision-chain information in C^clin.
    conflicts = [
        (("structure_visibility", "absent"), ("keyframe_status", "yes")),
        (("image_usability", "unacceptable"), ("recommended_action", "hold")),
        (("primary_defect", "severe_artifact"), ("recommended_action", "hold")),
        (("primary_defect", "thyroid_not_visible"), ("recommended_action", "hold")),
        (("primary_defect", "target_incomplete"), ("recommended_action", "hold")),
    ]
    for a, b in conflicts:
        if a in index and b in index:
            c[index[a], index[b]] = 1.0
            c[index[b], index[a]] = 1.0
    return c


def clinical_cost_matrix(tasks: Mapping[str, Sequence[str]], eta: float = 1.0) -> torch.Tensor:
    """Return C = C^sem + eta*C^clin."""
    return semantic_cost_matrix(tasks) + float(eta) * clinical_relation_cost_matrix(tasks)
