from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features=in_features,
                                           out_features=out_features),
                                 nn.ReLU(),
                                 #nn.LayerNorm(out_features)
                                 )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp(x)
        x = self.drop(x)
        return x


class Encoder(nn.Module):
    def __init__(self, features=98, layers=[64, 32, 16], dropout=0.1):
        super().__init__()

        encoder_layers = [features] + layers
        self.encoder = nn.Sequential(*[MLPBlock(f_in, f_out, dropout)
                                       for (f_in, f_out) in zip(encoder_layers, encoder_layers[1:])])

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, features=98, layers=[64, 32, 16], dropout=0.1):
        super().__init__()

        decoder_layers = list(reversed(layers)) + [features]
        self.decoder = nn.Sequential(*[MLPBlock(f_in, f_out, dropout)
                                       for (f_in, f_out) in zip(decoder_layers, decoder_layers[1:])])

    def forward(self, x):
        x = self.decoder(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, features=98, layers=[64, 32, 16], dropout=0.0):
        super().__init__()
        self.encoder = Encoder(features, layers, dropout)
        self.decoder = Decoder(features, layers, dropout)

    def forward(self, x):
        # print("AE: " + str(x.shape))
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#model = Autoencoder(features=98, layers=[64, 32, 16], dropout=0.1)
#model