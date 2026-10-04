"""CPU checks for the released MCC-OPD core; no models or datasets required."""

import pytest
import torch
from scipy.optimize import brentq

from mcc_opd import (
    build_union_groups,
    cf_mirror_coefficient,
    mcc_partition_loss,
    mcc_token_loss,
    partition_log_probs,
)


def test_mirror_budget_against_independent_root_solver():
    # Adapted from the training implementation's existing mirror-budget test.
    full = torch.tensor(
        [[.8, .15, .05], [.05, .15, .8], [.2, .3, .5]], dtype=torch.float64
    ).log().requires_grad_()
    delta = torch.tensor(
        [[0., 1., 3.], [0., 1., 3.], [4., 4., 4.]],
        dtype=torch.float64, requires_grad=True,
    )
    coefficients = cf_mirror_coefficient(full, delta)
    assert not coefficients.requires_grad
    assert (coefficients >= 0).all() and (coefficients <= 1).all()
    assert (coefficients < .99).any() and (coefficients == 1).any()
    torch.testing.assert_close(coefficients, cf_mirror_coefficient(full, delta + 37.))
    for f, d, c in zip(full, delta, coefficients):
        def kl(x):
            target = (f + x * d).log_softmax(-1)
            return (f.exp() * (f - target)).sum().item()

        budget = kl(-1.)
        expected = brentq(lambda x: kl(x) - budget, 0., 1.) if kl(1.) > budget + 1e-12 else 1.
        assert c.item() == pytest.approx(expected, abs=3e-10)
        assert kl(c) <= budget + 1e-12


def test_partition_target_and_teacher_are_detached():
    z = torch.tensor([.4, -.2, .6], dtype=torch.float64, requires_grad=True)
    full = torch.tensor([.8, .15, .05], dtype=torch.float64).log().requires_grad_()
    cf = torch.tensor([.05, .15, .8], dtype=torch.float64).log().requires_grad_()
    loss, coefficient = mcc_partition_loss(z.log_softmax(-1), full, cf)
    gradients = torch.autograd.grad(loss, (z, full, cf), allow_unused=True)
    assert torch.isfinite(loss) and torch.isfinite(gradients[0]).all()
    assert gradients[1:] == (None, None)
    assert not coefficient.requires_grad


def test_real_tail_has_no_floor_and_retains_rare_token_gradient():
    # Adapted from the existing true-tail gradient test.
    logits = torch.tensor([[[0., -30., -31.]]], dtype=torch.float64, requires_grad=True)
    lp = partition_log_probs(logits, torch.tensor([[[0]]]), output_dtype=torch.float64)
    assert lp.exp()[0, 0, -1] < 1e-12
    target = torch.tensor([.5, .5], dtype=torch.float64)
    loss = (target * (target.log() - lp)).sum()
    gradient = torch.autograd.grad(loss, logits)[0]
    tail_proportion = logits[0, 0, 1:].softmax(-1).detach()
    torch.testing.assert_close(gradient[0, 0, 1:], -.5 * tail_proportion, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(lp.exp().sum(-1), torch.ones((1, 1), dtype=torch.float64))


def test_union_is_deduplicated_and_grouped_by_actual_size():
    student_ids = torch.tensor([[[0, 1], [0, 1], [0, 1]]])
    teacher_ids = torch.tensor([[[0, 1], [1, 2], [2, 3]]])
    groups = build_union_groups(student_ids, teacher_ids)
    assert [g.token_ids.shape[-1] for g in groups] == [2, 3, 4]
    for group in groups:
        for position, ids in zip(group.positions.tolist(), group.token_ids[0].tolist()):
            expected = (set(student_ids[0, position].tolist())
                        | set(teacher_ids[0, position].tolist()))
            assert set(ids) == expected and len(ids) == len(expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("alpha", [0., 1., 1.5])
def test_token_loss_and_gradient_against_dense_probability_reference(dtype, alpha):
    rng = torch.Generator().manual_seed(45)
    student = torch.randn(1, 5, 13, generator=rng, dtype=dtype).requires_grad_()
    full = torch.randn(1, 5, 13, generator=rng, dtype=dtype).requires_grad_()
    cf = torch.randn(1, 5, 13, generator=rng, dtype=dtype).requires_grad_()
    losses, coefficients = mcc_token_loss(student, full, cf, topk=2, alpha=alpha, chunk_size=2)
    actual_grad = torch.autograd.grad(losses.sum(), (student, full, cf), allow_unused=True)
    assert actual_grad[1:] == (None, None)
    assert not coefficients.requires_grad

    reference_student = student.detach().float().double().requires_grad_()
    p = reference_student.softmax(-1)
    q_full = full.detach().double().softmax(-1)
    q_cf = cf.detach().double().softmax(-1)
    expected_losses, expected_coefficients = [], []
    for t in range(student.shape[1]):
        # Python sets and explicit probability sums independently define each partition.
        ids = sorted(set(student[0, t].topk(2).indices.tolist()) |
                     set(full[0, t].topk(2).indices.tolist()))
        outside = [i for i in range(student.shape[-1]) if i not in ids]

        def aggregate(probabilities):
            return torch.cat((probabilities[0, t, ids], probabilities[0, t, outside].sum()[None]))

        student_p = aggregate(p)
        full_p, cf_p = aggregate(q_full), aggregate(q_cf)
        base = full_p.log()
        direction = alpha * (base - cf_p.log())

        def kl(c):
            target = (base + c * direction).log_softmax(-1)
            return (full_p * (base - target)).sum().item()

        budget = kl(-1.)
        c = brentq(lambda x: kl(x) - budget, 0., 1.) if kl(1.) > budget + 1e-12 else 1.
        target = (base + c * direction).log_softmax(-1)
        expected_coefficients.append(c)
        expected_losses.append((student_p * (student_p.log() - target)).sum())
    expected = torch.stack(expected_losses)[None]
    reference_grad = torch.autograd.grad(expected.sum(), reference_student)[0]
    torch.testing.assert_close(losses.double(), expected, atol=8e-7, rtol=2e-6)
    torch.testing.assert_close(
        coefficients, torch.tensor([expected_coefficients], dtype=torch.float64),
        atol=3e-6, rtol=3e-6,
    )
    gradient_tolerance = 3e-3 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(
        actual_grad[0].double(), reference_grad, atol=gradient_tolerance, rtol=gradient_tolerance
    )


def test_chunking_preserves_loss_and_gradient():
    rng = torch.Generator().manual_seed(7)
    tensors = [torch.randn(1, 7, 17, generator=rng) for _ in range(3)]
    student = tensors[0].requires_grad_()
    one, c_one = mcc_token_loss(student, *tensors[1:], topk=3, chunk_size=1)
    many, c_many = mcc_token_loss(student, *tensors[1:], topk=3, chunk_size=128)
    torch.testing.assert_close(one, many)
    torch.testing.assert_close(c_one, c_many)
    g_one = torch.autograd.grad(one.sum(), student)[0]
    g_many = torch.autograd.grad(many.sum(), student)[0]
    torch.testing.assert_close(g_one, g_many)
