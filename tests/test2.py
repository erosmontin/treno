# https://idiomaticprogrammers.com/post/unet-architecture/
import torch
from treno.models import UNet

from torchvision.transforms import transforms
from tqdm import tqdm
import torchvision

from torch.utils.data import DataLoader
import torch

from torch.utils.data import Dataset
from PIL import Image
import os
class BirdDataset(Dataset):
    def __getitem__(self, index):
        image_name = ".".join(self.images_paths[index].split('.')[:-1])

        image = Image.open(os.path.join(self.image_dir, f"{image_name}.jpg")).convert("RGB")
        seg = Image.open(os.path.join(self.segmentation_dir, f"{image_name}.png")).convert("L")

        image = self.transform_image(image)
        seg = self.transform_mask(seg)

        return image, seg

    def __init__(self, image_paths, image_dir, segmentation_dir, transform_image, transform_mask):
        super(BirdDataset, self).__init__()
        self.image_dir = image_dir
        self.segmentation_dir = segmentation_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        with open(image_paths, 'r') as f:
            self.images_paths = [line.split(" ")[-1] for line in f.readlines()]

    def __len__(self):
        return len(self.images_paths)

def load_data_set(image_paths, image_dir, segmentation_dir, transforms, batch_size=8, shuffle=True):
    dataset = BirdDataset(image_paths,
                          image_dir,
                          segmentation_dir,
                          transform_image=transforms[0],
                          transform_mask=transforms[1])

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [19, 1])

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    ), DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=shuffle
        )


config = {
    "lr": 1e-3,
    "batch_size": 2,
    "image_dir": "/data/MYDATA/DL_train_DS/birds/CUB_200_2011/images",
    "segmentation_dir": "/data/MYDATA/DL_train_DS/birds/segmentations",
    "image_paths": "/data/MYDATA/DL_train_DS/birds/CUB_200_2011/images_cut.txt",
    "epochs": 10,
    "checkpoint": "/data/garbage/bird_segmentation_v1.pth",
    "optimiser": "/data/garbage/bird_segmentation_v1_optim.pth",
    "continue_train": False,
    # "device": "cuda" if torch.cuda.is_available() else "cpu"
    "device": "cpu"
}


transforms_image = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0., 0., 0.), (1., 1., 1.))
])

transforms_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.,), (1.,))
])

train_dataset, val_dataset = load_data_set(
    config['image_paths'],
    config['image_dir'],
    config['segmentation_dir'],
    transforms=[transforms_image, transforms_mask],
    batch_size=config['batch_size']
)

print("loaded", len(train_dataset), "batches")

ff=[4,8,16,32]
pre=f'{ff[0]}_'
model = UNet(3,features=ff).to(config['device'])
optimiser = torch.optim.Adam(params=model.parameters(), lr=config['lr'])

if config['continue_train']:
    state_dict = torch.load(config['checkpoint'])
    optimiser_state = torch.load(config['optimiser'])
    model.load_state_dict(state_dict)
    optimiser.load_state_dict(optimiser_state)

loss_fn = torch.nn.BCEWithLogitsLoss()

model.train()
def check_accuracy_and_save(model, optimiser, epoch):
    torch.save(model.state_dict(), config['checkpoint'])
    torch.save(optimiser.state_dict(), config['optimiser'])

    dice_score = 0

    model.eval()
    with torch.no_grad():
        for x, y in val_dataset:
            x = x.to(config['device'])
            y = y.to(config['device'])

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

            torchvision.utils.save_image(preds, f"test/pred/{epoch}.png")
            torchvision.utils.save_image(y, f"test/true/{epoch}.png")

    print(
        f"Dice Score = {dice_score/len(val_dataset)}"
    )
    model.train()
from pyable_eros_montin import imaginable as ima

import numpy as np
def train():
    for epoch in range(config['epochs']):
        loop = tqdm(train_dataset)
        for image, seg in loop:
            image = image.to(config['device'])
            seg = seg.float().to(config['device'])


            pred = model(image)
            loss = loss_fn(pred, seg)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            loop.set_postfix(loss=loss.item())
        im,_=next(iter(val_dataset))
        im,_=next(iter(val_dataset))
        preds = torch.sigmoid(model(im))
        ima.saveNumpy(preds[0,0].cpu().detach().numpy().astype(np.float32),f'/data/garbage/{pre}s{epoch}.nii.gz')
        # check_accuracy_and_save(model, optimiser, epoch)

train()