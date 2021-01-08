from effdet_mmdet.models import BiFPN
from effdet_mmdet.models.necks.utils import WeightedAdd
import torch


def test_neck():
    bs = 1
    out_channels = 64

    feats = [
        torch.rand([bs, 40, 128, 128]),
        torch.rand([bs, 112, 64, 64]),
        torch.rand([bs, 320, 32, 32])]

    neck = BiFPN(
        in_channels_list=[40, 112, 320],
        out_channels=64,
        stack=3)

    feats = neck(feats)
    feats_shapes = [f.shape for f in feats]

    truth_shapes = [(bs, out_channels, 128, 128),
                    (bs, out_channels, 64, 64),
                    (bs, out_channels, 32, 32),
                    (bs, out_channels, 16, 16),
                    (bs, out_channels, 8, 8)]
    assert all([fs == ts for fs, ts in zip(feats_shapes, truth_shapes)])
    

class TestWAdd:
    def test_shape(self):
        wadd = WeightedAdd(num_inputs=2)
        x = [torch.rand(1, 64, 16, 16), torch.rand(1, 64, 16, 16)]
        output = wadd(x)
        assert output.shape == (1, 64, 16, 16)

    def test_output_and_weight(self):
        wadd = WeightedAdd(num_inputs=3, eps=1e-8)
        x = [torch.rand(1, 64, 16, 16) * i for i in range(1, 4)]
        with torch.no_grad():
            output = wadd(x)
        truth_output = torch.cat(x, dim=0).mean(dim=0)

        assert torch.allclose(output, truth_output, atol=1e-3), \
            f"difference: {torch.abs(output - truth_output).sum()}"
        assert torch.allclose(wadd.weights, torch.full((3,), 1 / 3))