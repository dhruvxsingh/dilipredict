from dilipredict import models, pipelines


def test_MutSmiReg(image_loader):
    test_img_path = 'tests/data/test_img.tiff'
    image_files = [
        [test_img_path] * 10,
        [test_img_path] * 10,
        [test_img_path] * 10,
    ]

    model = models.DILIPredict()
    pipeline = pipelines.DILIPredict(image_loader, model)
    out, prob = pipeline(image_files)
    assert isinstance(out, int)
    assert out in [i for i in range(3)]
    assert isinstance(prob, list)
    assert isinstance(prob[0], float)
