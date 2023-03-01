import torch
import torch.nn as nn

import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    '''
    PE(pos, 2i)     = sin(pos/(10000^(2i/embSize)))
    PE(pos, 2i+1)   = cos(pos/(10000^(2i/embSize)))
    pos is the position in the sequence
    i is the dimension, aka embedding size, embSize

    Remember
    e^(ln(x)) = x
    ln(x^z) = z ln(x)

    10000^(2i/embSize) = e^((2i/embSize) * ln(10000))

    dropout applied to sum of embedding and positional encoding
    '''
    def __init__(self, embSize, device, dropout=0.1, maxLen=1024):
        super().__init__()
        denominator = torch.exp(torch.arange(0, embSize, 2) * torch.log(torch.tensor(10000)) / embSize)
        pos = torch.arange(0, maxLen).reshape(maxLen, 1)

        pe = torch.zeros((maxLen, embSize), dtype=torch.float)
        pe[:, 0::2] = torch.sin(pos / denominator)
        pe[:, 1::2] = torch.cos(pos / denominator)
        pe.requires_grad = False

        self.pe = pe.unsqueeze(0).to(device)
        self.dropout = nn.Dropout(dropout)
        self.embSize = embSize
        self.maxLen = maxLen

    def forward(self, emb):
        return self.dropout(emb + self.pe[:,:emb.shape[1],:])

class PositionWiseFeedforward(nn.Module):
    def __init__(self, embSize, pwffDim):
        super().__init__()
        self.fc1 = nn.Linear(embSize, pwffDim)
        self.fc2 = nn.Linear(pwffDim, embSize)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, embSize, nHeads):
        super().__init__()
        assert embSize % nHeads == 0

        self.embSize = embSize
        self.nHeads = nHeads
        self.headDim = embSize // nHeads

        self.fcQ = nn.Linear(embSize, embSize)
        self.fcK = nn.Linear(embSize, embSize)
        self.fcV = nn.Linear(embSize, embSize)

        self.fcOut = nn.Linear(embSize, embSize)

        self.scale = torch.sqrt(torch.tensor(self.embSize, dtype=torch.float))

    def forward(self, query, key, value, mask=None):
        batchSize = query.shape[0]

        Q = self.fcQ(query)
        K = self.fcK(key)
        V = self.fcV(value)

        Q = Q.view(batchSize, -1, self.nHeads, self.headDim).permute(0, 2, 1, 3)
        K = K.view(batchSize, -1, self.nHeads, self.headDim).permute(0, 2, 1, 3)
        V = V.view(batchSize, -1, self.nHeads, self.headDim).permute(0, 2, 1, 3)

        innerProd = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            innerProd = innerProd.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(innerProd, dim=-1)

        out = torch.matmul(attention, V)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batchSize, -1, self.embSize)
        out = self.fcOut(out)

        return out, attention

class Block(nn.Module):
    def __init__(self, embSize, nHeads, pwffDim, dropout):
        super().__init__()
        self.maskedSelfAttention = MultiHeadAttention(embSize, nHeads)
        self.maskedAttentionLayerNorm = nn.LayerNorm(embSize)

        self.pwff = PositionWiseFeedforward(embSize, pwffDim)
        self.pwffLayerNorm = nn.LayerNorm(embSize)

        self.dropout = nn.Dropout(dropout)

    def forward(self, emb, mask=None):
        out, maskedAttention = self.maskedSelfAttention(emb, emb, emb, mask)
        emb = self.maskedAttentionLayerNorm(emb + self.dropout(out))

        out = self.pwff(emb)
        emb = self.pwffLayerNorm(emb + self.dropout(out))

        return emb, maskedAttention

class YelpModel(nn.Module):
    def __init__(self, nLayers, embSize, vocabSize, device, 
                 nHeads=8, pwffDim=512, dropout=0.1):
        super().__init__()
        self.nLayers = nLayers
        self.embSize = embSize

        self.emb = nn.Embedding(vocabSize, embSize)
        self.pe = PositionalEncoding(embSize, dropout=dropout, device=device)

        self.blocks = nn.ModuleList([Block(embSize, nHeads, pwffDim, dropout) for _ in range(nLayers)])
        self.fcOut = nn.Linear(embSize, vocabSize)

    def forward(self, x, mask=None):
        x = self.pe(self.emb(x))

        for block in self.blocks:
            x, attn = block(x, mask)
        
        out = self.fcOut(x)
        return out

def createMask(vec, padIdx, diagnol=0):
    padMask = torch.ones(vec.shape).long()
    padMask[vec == padIdx] = 0
    padMask = padMask.unsqueeze(1).unsqueeze(1)

    seqLen = vec.shape[1]
    #diagnol is how much you offset the mask
    #1 moves it to the right
    seqMask = torch.tril(torch.ones((seqLen, seqLen)), diagnol).long()
    seqMask = seqMask.unsqueeze(0).unsqueeze(0)

    mask = padMask & seqMask

    return mask


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    vocabSize = 50000 
    seqLen = 512
    nLayers = 4
    embSize = 256
    batchSize = 16

    x = torch.randint(0, vocabSize, (batchSize, seqLen)).to(device)
    x[:,seqLen - 3:] = 0 #pad it

    mask = createMask(x, 0)
    mask = mask.to(device)

    model = YelpModel(nLayers, embSize, vocabSize, device).to(device)

    y = model(x, mask)

    print(y.shape) 
    return

if __name__ == '__main__':
    main()


