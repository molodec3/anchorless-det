import torch

from model import CenterNet
from face_dataset.dataset import CenterFaceDataset, center_face_train_test_split
from loss import center_net_loss
from torch.utils.data import DataLoader


def train(batch_size=32, epochs=100):
    train_files, test_files, df = center_face_train_test_split(
        helen_path='./data/helen/helen_1',
        fgnet_path='./data/fg_net/images',
        celeba_path='./data/celeba/img_align_celeba',
    )

    train_dataset = CenterFaceDataset(train_files, df)
    test_dataset = CenterFaceDataset(test_files, df)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = CenterNet(train_dataset.n_classes)
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        for img, mask, size in train_dataloader:
            img = img.to(device)
            mask = mask.to(device)
            size = size.to(device)

            pred_hm, pred_offset, pred_size = model(img)

            loss = center_net_loss(
                pred_hm, pred_offset, pred_size,
                mask, size
            )
            # TODO a lot

        model.eval()


if __name__ == '__main__':
    pass
