import torch
import torch.nn as nn
from dilipredict.image_loader import ImageLoader


class DILIPredict:
    def __init__(self, image_loader: ImageLoader, model: nn.Module, device: torch.device = torch.device('cpu')) -> None:
        self.image_loader = image_loader
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        self.model = model
        self.device = device

    def __call__(self, img_dir: str) -> int:
        '''
        img_dir: a sample image file dir
        '''
        img_x = self.image_loader.encoder(img_dir).unsqueeze(0)
        out = self.model(img_x)
        out_label = out.argmax(dim=-1).item()
        probability = out.cpu().numpy().tolist()[0]
        return out_label, probability
