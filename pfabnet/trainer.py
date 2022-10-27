from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import pickle


class TrainerConfig:
    # optimization parameters
    betas = (0.9, 0.999)
    grad_norm_clip = 1.0
    ckpt_path = None
    history_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model.state_dict(), self.config.ckpt_path)


    def train(self):
        model, config = self.model, self.config
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                      betas=config.betas, weight_decay=0.005)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)

            if is_train:
               data = self.train_dataset
               batch_size = config.batch_size
            else:
               data = self.val_dataset
               batch_size = config.batch_size

            shuffle = False
            if is_train:
               shuffle = True

            loader = DataLoader(data, shuffle=shuffle, pin_memory=True,
                                batch_size=batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, d_it in pbar:
                x, y = d_it

                x = x.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    output, loss = model(x, y)
                    loss = loss.mean() 
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. "
                                         f"lr {config.learning_rate:e}")

            return float(np.mean(losses))

        best_loss = float('inf')
        try:
            with open(self.config.history_path, 'rb') as fptr:
                history = pickle.load(fptr)
                start_epoch = len(np.array(history['val_loss']))
                history = {'train_loss':history['train_loss'][:start_epoch], 'val_loss':history['val_loss'][:start_epoch]}
        except Exception as e:
            history = {'train_loss': [], 'val_loss': []}
            start_epoch = 0

        for epoch in range(start_epoch, config.max_epochs):
            train_loss = run_epoch('train')
            val_loss = run_epoch('val')
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            with open(self.config.history_path, 'wb') as fptr:
                pickle.dump(history, fptr)

            if epoch < 1950:
                self.save_checkpoint()
                continue

            good_model = val_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = val_loss
                self.save_checkpoint()


