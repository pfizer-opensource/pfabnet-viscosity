import torch
import torch.nn as nn


class ViscosityNet(nn.Module):
    def __init__(self, num_channels=2):
        super(ViscosityNet, self).__init__()
        nfilt = num_channels
        ks = 3

        dilation = 1
        if num_channels == 2:
            self.convnet = nn.Sequential(nn.Conv3d(num_channels, nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
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
                                         nn.Conv3d(32*nfilt, 1024, ks, padding='same', dilation=dilation), nn.ReLU()
                                         )
        elif num_channels == 3:
            self.convnet = nn.Sequential(nn.Conv3d(num_channels, nfilt, ks, padding='same', dilation=dilation), nn.ReLU(),
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
                                         nn.Conv3d(32*nfilt, 1024, ks, padding='same', dilation=dilation), nn.ReLU()
                                         )
        else:
            print('ERROR... number of input channels must be either 2 or 3')


        self.fc = nn.Sequential(nn.Linear(1024, 1), nn.ReLU())

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
            return x, emb

