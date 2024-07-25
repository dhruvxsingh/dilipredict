import pytest
from dilipredict import nets, datasets


@pytest.fixture
def stvit_conf():
    return nets.STViTConfig(
        dim=768,
        num_classes=3,
        pool='mean',
        img_encoder_conf=nets.beitv2.BeitV2.DEFAULT_CONF
    )

@pytest.fixture
def stvit_ds(image_loader):
    test_img_path = 'tests/data/test_img.tiff'
    sample_image_files = [
        [test_img_path] * 10,
        [test_img_path] * 10,
        [test_img_path] * 10,
    ]
    ds = datasets.DILIPredict(image_loader)
    img_x = ds(sample_image_files)
    return img_x.unsqueeze(0)

def test_STViT(stvit_conf, stvit_ds):

    model = nets.STViT(stvit_conf)
    out = model(stvit_ds)
    assert out.dim() == 2
    assert out.size(1) == stvit_conf.num_classes
