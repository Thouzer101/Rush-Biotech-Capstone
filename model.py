import torch
import torch.nn as nn

import numpy as np
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
    def __init__(self, embSize, maxLen, device):
        super().__init__()
        denominator = torch.exp(torch.arange(0, embSize, 2) * torch.log(torch.tensor(10000)) / embSize)
        pos = torch.arange(0, maxLen).reshape(maxLen, 1)

        pe = torch.zeros((maxLen, embSize), dtype=torch.float)
        pe[:, 0::2] = torch.sin(pos / denominator)
        pe[:, 1::2] = torch.cos(pos / denominator)
        pe.requires_grad = False

        self.pe = pe.unsqueeze(0)
        self.embSize = embSize
        self.maxLen = maxLen
        self.device = device

    def forward(self, tok):
        #We do not want to add
        #we want to be able to predict the position in the future
        batchSize = tok.shape[0]
        peTen = self.pe.repeat(batchSize, 1, 1).to(self.device)
        out = peTen[torch.arange(batchSize).unsqueeze(1), tok]

        return out

class NonContinuousPositionalEncoding(nn.Module):
    def __init__(self, embSize, maxLen, device):
        super().__init__()
        self.pe = nn.Embedding(maxLen, embSize)
        self.embSize = embSize
        self.maxLen = maxLen
        self.device = device
    
    def forward(self, tok):
        out = self.pe(tok)
        return out

class PositionWiseFeedfoward(nn.Module):
    def __init__(self, embSize, pwffDim):
        super().__init__()
        self.fc1 = nn.Linear(embSize, pwffDim)
        self.fc2 = nn.Linear(pwffDim, embSize)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MultiHeadAttention(nn.Module):
    '''
    Attention(Q, K, V) = softmax((Q x transpose(K)) / sqrt(embSize)) x V
    embSize is split up into nHeads since softmax mainly
    attents to one thing at a time

    Pass query, key, value through linear layers
    Do scaled dot-product attention
    concat
    final linear layer
    '''
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

        '''
        Linearly project to dq, dk, and dv
        go from [batchSize, seqLen, embSize]
        to      [batchSize, seqLen, nHeads, headDim]
        then permute
        to      [batchSize, nHeads, seqLen, headDim]
        '''
        Q = Q.view(batchSize, -1, self.nHeads, self.headDim).permute(0, 2, 1, 3)
        K = K.view(batchSize, -1, self.nHeads, self.headDim).permute(0, 2, 1, 3)
        V = V.view(batchSize, -1, self.nHeads, self.headDim).permute(0, 2, 1, 3)

        '''
        Inner product of Q and K
        Basically how similiar they are
        Q = [batchSize, nHeads, seqLenQuery, headDim]
        K = [batchSize, nHeads, headDim, seqLenKey]
        output becomes [batchSize, nHeads, seqLenQuery, seqLenKey]
        '''
        innerProd = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            innerProd = innerProd.masked_fill(mask == 0, float('-inf'))

        #softmax along last dimension, the columns
        attention = torch.softmax(innerProd, dim=-1)

        #which part of values should be paid attention to
        #Attention = [batchSize, nHeads, seqLen, seqLen]
        #V = [batchSize, nHeads, seqLen, headDim]
        #out = [batchSize, nHeads, seqLen, headDim]
        out = torch.matmul(attention, V)

        #back to [batchSize, seqLen, nHeads, headDim]
        out = out.permute(0, 2, 1, 3).contiguous()
        #back to [batchSize, seqLen, embSize]
        out = out.view(batchSize, -1, self.embSize)

        out = self.fcOut(out)

        return out, attention

        

class Block(nn.Module):
    def __init__(self, embSize, nHeads, pwffDim, dropout):
        super().__init__()
        self.maskedSelfAttention = MultiHeadAttention(embSize, nHeads)
        self.maskedAttentionLayerNorm = nn.LayerNorm(embSize)

        self.pwff = PositionWiseFeedfoward(embSize, pwffDim)
        self.pwffLayerNorm = nn.LayerNorm(embSize)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, emb, mask=None):
        out, maskedAttention = self.maskedSelfAttention(emb, emb, emb, mask)
        emb = self.maskedAttentionLayerNorm(emb + self.dropout(out))

        out = self.pwff(emb)
        emb = self.pwffLayerNorm(emb + self.dropout(out))

        return emb, maskedAttention

class MimicModel(nn.Module):
    def __init__(self, nLayers, embSize, nTokens, peSize, device,
                 maxHrs=2048, nCls=1, nHeads=8, pwffDim=512, dropout=0.1):
        super().__init__()
        self.nLayers = nLayers
        self.embSize = embSize

        #self.pe = PositionalEncoding(peSize, maxHrs, device)
        self.pe = NonContinuousPositionalEncoding(peSize, maxHrs, device)
        self.em = nn.Embedding(nTokens, embSize - peSize - 1)

        self.blocks = nn.ModuleList([Block(embSize, nHeads, pwffDim, dropout)
                                     for _ in range(nLayers)])

        #for unsuperivsed pretraining 
        #output indices 
        self.fcIdx = nn.Linear(embSize, nTokens)
        #output vals
        self.fcVals = nn.Linear(embSize, 1)
        #output times
        self.fcTimes = nn.Linear(embSize, maxHrs)

        #predict cls
        self.fcCls = nn.Linear(embSize, nCls)

    def forward(self, tokTen, valsTen, timesTen, mask=None):
        tokEmb = self.em(tokTen)
        timesEmb = self.pe(timesTen)

        emb = torch.cat((tokEmb, valsTen.unsqueeze(-1), timesEmb), dim=-1)

        for layer in self.blocks:
            emb, attention = layer(emb, mask)

        return emb
    
    def unsupervised(self, emb):
        #use the output of forward
        idxOut = self.fcIdx(emb)
        valsOut = self.fcVals(emb)
        timesOut = self.fcTimes(emb)

        return idxOut, valsOut, timesOut
    
    def supervised(self, emb):
        #use the output of forward

        #just first token?
        emb = emb[:,0,:]
        out = self.fcCls(emb)
        return out

def createMask(tokTen, padIdx):
    padMask = torch.ones(tokTen.shape).long()
    padMask[tokTen == padIdx] = 0
    padMask = padMask.unsqueeze(1).unsqueeze(1)

    seqLen = tokTen.shape[1]
    seqMask = torch.tril(torch.ones((seqLen, seqLen))).long() 
    seqMask = seqMask.unsqueeze(0).unsqueeze(0)

    mask = padMask & seqMask

    return mask

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batchSize = 16

    seqLen = 1024
    maxHrs = 24 * 60
    
    nLayers = 4
    nTokens = 3200
    embSize = 256
    peSize = 64
    dropout = 0.1
    padIdx = 0
    clsIdx = 1

    tokTen = torch.randint(0, nTokens, (batchSize, seqLen)).long()
    tokVals = torch.randn(batchSize, seqLen).float()
    tokTimes = torch.randint(0, maxHrs, (batchSize, seqLen)).long()


    mask = createMask(tokTen, padIdx).to(device)

    model = MimicModel(nLayers, embSize, nTokens, peSize, device).to(device)


    #clsIdx, demoIdx, eventIndices, eventVals, eventTimes
    tokTen = tokTen.to(device)
    tokVals = tokVals.to(device)
    tokTimes = tokTimes.to(device)
    
    out = model(tokTen, tokVals, tokTimes, mask)

    model.unsupervised(out)
    #model.supervised(rawOut)
    return

if __name__ == '__main__':
    main()