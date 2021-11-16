import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import CenterNet
from .face_dataset.dataset import CenterFaceDataset, center_face_train_test_split
from .loss import center_net_loss
from .utils import make_bin_mask, calculate_iou


def train_cycle(
    model, optimizer, 
    train_data_gen, val_data_gen,# n_classes,
    epochs=100, name='centernet'
):
    train_loss = []
    val_loss = []
    val_metric = []
    
    best_metric = 0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.device = device

    total_time = time.time()
    for epoch in range(epochs):
        train_epoch_loss = 0
        val_epoch_loss = 0
        val_epoch_metric = 0
        
        train_count = 0
        val_count = 0
        
        model.train()
        start = time.time()
        for img, mask, size in tqdm(train_data_gen):
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
        print(f'Epoch: {epoch + 1} of {epochs}\ntrain loss: {train_loss[-1]}\n'
              f'train time spent: {time.time() - start:.2f}')

        start = time.time()
        model.eval()
        with torch.no_grad():
            for img, mask, size, bin_mask in tqdm(val_data_gen):
                img = img.to(device)
                mask = mask.to(device)
                size = size.to(device)
                bin_mask = bin_mask.to(device)
            
                pred_hm, pred_offset, pred_size = model(img)
                loss = center_net_loss(
                    pred_hm, pred_offset, pred_size,
                    mask, size
                )
            
                pred = model.predict(img)  # make_bin_mask might not work due to different devices
                iou = calculate_iou(bin_mask, make_bin_mask(pred, bin_mask.shape).to(device))
                
                val_epoch_loss += loss.cpu().data.numpy()
                val_epoch_metric += iou.cpu().data.numpy()
                val_count += 1
                
            val_loss.append(val_epoch_loss / val_count)
            val_metric.append(val_epoch_metric / val_count)
            print(f'val_loss: {val_loss[-1]}\nval_metric: {val_metric[-1]}\n'
                  f'val time spent: {time.time() - start:.2f}')
            
            if val_metric[-1] > best_metric:
                best_metric = val_metric[-1]
                torch.save(model.state_dict(), f'{name}.pth')
                print('Current model saved.\n')

    print(f'Total time spent: {time.time() - total_time:.2f}')
            
    return train_loss, val_loss, val_metric


if __name__ == '__main__':
    pass
