import torch
from torch.utils.data import DataLoader

from .model import CenterNet
from .face_dataset.dataset import CenterFaceDataset, center_face_train_test_split
from .loss import center_net_loss
from .utils import make_bin_mask, calculate_iou


def train_cycle(
    model, optimizer, 
    train_data_gen, val_data_gen,# n_classes,
    epochs=100
):
    train_loss = []
    val_loss = []
    val_metric = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.device = device

    for epoch in range(epochs):
        train_epoch_loss = 0
        val_epoch_loss = 0
        val_epoch_metric = 0
        
        train_count = 0
        val_count = 0
        
        model.train()
        for img, mask, size in train_data_gen:
            img = img.to(device)
            mask = mask.to(device)
            size = size.to(device)
            optimizer.zero_grad()
            
            pred_hm, pred_offset, pred_size = model(img)
            loss = center_net_loss(
                pred_hm, pred_offset, pred_size,
                mask, size
            )
            
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.cpu().data.numpy()
            train_count += 1
            
        train_loss.append(train_epoch_loss / train_count)
        print(f'Epoch: {epoch}\ntrain loss: {train_loss[-1]}\n')

        model.eval()
        with torch.no_grad():
            for img, mask, size, bin_mask in val_data_gen:
                img = img.to(device)
                mask = mask.to(device)
                size = size.to(device)
                bin_mask = bin_mask.to(device)
            
                pred_hm, pred_offset, pred_size = model(img)
                loss = center_net_loss(
                    pred_hm, pred_offset, pred_size,
                    mask, size
                )
            
                pred = model.predict(img)
                iou = calculate_iou(bin_mask, make_bin_mask(pred, bin_mask.shape))
                
                val_epoch_loss += loss.cpu().data.numpy()
                val_epoch_metric += iou.cpu().data.numpy()
                val_count += 1
                
            val_loss.append(val_epoch_loss / val_count)
            val_metric.append(val_epoch_metric / val_count)
            print(f'val_loss: {val_loss[-1]}\nval_metric: {val_metric[-1]}\n')
            
    return train_loss, val_loss, val_metric


if __name__ == '__main__':
    pass
