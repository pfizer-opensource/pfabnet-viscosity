import torch
import torch.nn as nn


class ViscosityNet(nn.Module):
    def __init__(self, grid_dim=96):
        super(ViscosityNet, self).__init__()
        nfilt = 2
        ks = 3

        dilation = 1
        if grid_dim >= 64:
            self.convnet = nn.Sequential(nn.Conv3d(1, nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
                                         nn.MaxPool3d(2),
                                         nn.Conv3d(nfilt, 2*nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
                                         nn.MaxPool3d(2),
                                         nn.Conv3d(2*nfilt, 4*nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
                                         nn.MaxPool3d(2),
                                         nn.Conv3d(4*nfilt, 8*nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
                                         nn.MaxPool3d(2),
                                         nn.Conv3d(8*nfilt, 16*nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
                                         nn.MaxPool3d(2),
                                         nn.Conv3d(16*nfilt, 32*nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
                                         nn.MaxPool3d(2),
                                         nn.Conv3d(32*nfilt, 512*nfilt, ks, padding='same', dilation=dilation), nn.ReLU()
                                         )
        else:
            self.convnet = nn.Sequential(nn.Conv3d(1, nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
                                         nn.MaxPool3d(2),
                                         nn.Conv3d(nfilt, 2*nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
                                         nn.MaxPool3d(2),
                                         nn.Conv3d(2*nfilt, 4*nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
                                         nn.MaxPool3d(2),
                                         nn.Conv3d(4*nfilt, 8*nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
                                         nn.MaxPool3d(2),
                                         nn.Conv3d(8*nfilt, 16*nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
                                         nn.MaxPool3d(2),
                                         nn.Conv3d(16*nfilt, 32*nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
                                         nn.Conv3d(32*nfilt, 512*nfilt, ks, padding='same', dilation=dilation), nn.ReLU()
                                         )


        self.fc = nn.Sequential(nn.Linear(512*nfilt, 1), nn.ReLU())

        self.drop_out = nn.Dropout(0.05)


    def forward(self, x, y=None):
        x = self.convnet(x)

        emb = torch.flatten(x, 1)

        x = self.drop_out(emb)
        x = self.fc(x)
       
        if y is not None:
            loss = nn.functional.huber_loss(y, x, reduction='mean')
            return x, loss
        else:
            return x

