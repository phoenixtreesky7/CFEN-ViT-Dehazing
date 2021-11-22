import actnorm
import pytest
import torch


def test_actnorm():
    an = actnorm.ActNorm1d(10)
    x = torch.distributions.Exponential(
        (1 + torch.arange(10)).float() / 10
    ).sample([100])
    assert (an(x).mean(0).abs() < 1e-5).all()
    assert ((an(x).std(0, unbiased=False) - 1).abs() < 1e-5).all()


def test_reset():
    an = actnorm.ActNorm1d(10)
    x = torch.distributions.Exponential(
        (1 + torch.arange(10)).float() / 10
    ).sample([100])
    an(x)
    y = x + 2 / an.scale
    assert (an(y).mean(0) > 1).all()
    assert (an.reset_()(y).mean(0).abs() < 1e-5).all()


def test_actnorm1d():
    an = actnorm.ActNorm1d(3)
    x = torch.randn(5, 3)
    an(x)
    assert (
        actnorm.ActNorm1d(3)(x.unsqueeze(2)).squeeze(2)
        == actnorm.ActNorm1d(3)(x)
    ).all()
    with pytest.raises(ValueError, match="expected 2D or 3D input"):
        an(x[0])
    with pytest.raises(ValueError, match="expected 2D or 3D input"):
        an(x[..., None, None])
    with pytest.raises(ValueError, match="expected 2D or 3D input"):
        an(x[..., None, None, None])


def test_actnorm2d():
    an = actnorm.ActNorm2d(3)
    x = torch.randn(5, 3, 10, 10)
    an(x)
    assert (
        actnorm.ActNorm2d(3)(x[:, :, :1, :1]).squeeze()
        == actnorm.ActNorm1d(3)(x[:, :, 0, 0])
    ).all()
    with pytest.raises(ValueError, match="expected 4D input"):
        an(x[0, :, 0, 0])
    with pytest.raises(ValueError, match="expected 4D input"):
        an(x[:, :, 0, 0])
    with pytest.raises(ValueError, match="expected 4D input"):
        an(x[:, :, :, 0])
    with pytest.raises(ValueError, match="expected 4D input"):
        an(x[:, :, None])


def test_actnorm3d():
    an = actnorm.ActNorm3d(3)
    x = torch.randn(5, 3, 10, 10, 10)
    an(x)
    assert (
        actnorm.ActNorm3d(3)(x[:, :, :1]).squeeze()
        == actnorm.ActNorm2d(3)(x[:, :, 0])
    ).all()
    with pytest.raises(ValueError, match="expected 5D input"):
        an(x[0, :, 0, 0, 0])
    with pytest.raises(ValueError, match="expected 5D input"):
        an(x[:, :, 0, 0, 0])
    with pytest.raises(ValueError, match="expected 5D input"):
        an(x[:, :, :, 0, 0])
    with pytest.raises(ValueError, match="expected 5D input"):
        an(x[:, :, 0, :, :])
