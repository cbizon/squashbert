import torch

from squashbert.model import SquashMLP, cosine_loss


def test_mlp_output_shape_and_norm():
    m = SquashMLP(n_hops=2, embed_dim=16, hidden=32)
    x = torch.randn(4, (2 * 2 + 1) * 16)
    y = m(x)
    assert y.shape == (4, 16)
    norms = y.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)


def test_cosine_loss_zero_for_identical_vectors():
    y = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    loss = cosine_loss(y, y)
    assert abs(loss.item()) < 1e-5


def test_cosine_loss_positive_for_orthogonal():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0]])
    loss = cosine_loss(a, b)
    assert abs(loss.item() - 1.0) < 1e-5


def test_mlp_can_overfit_tiny_dataset():
    """End-to-end wiring check: a few gradient steps should drive loss down."""
    torch.manual_seed(0)
    m = SquashMLP(n_hops=1, embed_dim=8, hidden=16)
    x = torch.randn(8, 3 * 8)
    y = torch.nn.functional.normalize(torch.randn(8, 8), dim=1)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    losses = []
    for _ in range(100):
        pred = m(x)
        loss = cosine_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.3
