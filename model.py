import torch
import torch.nn as nn

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
    def __init__(self, embSize, maxLen=2048):
        super().__init__()
        denominator = torch.exp(torch.arange(0, embSize, 2) * torch.log(torch.tensor(10000)) / embSize)
        pos = torch.arange(0, maxLen).reshape(maxLen, 1)

        pe = torch.zeros((maxLen, embSize), dtype=torch.float)
        pe[:, 0::2] = torch.sin(pos / denominator)
        pe[:, 1::2] = torch.cos(pos / denominator)
        pe.requires_grad = False

        self.pe = pe
        self.embSize = embSize
        self.maxLen = maxLen

    def forward(self, emb):
        batchSize, seqLen = emb.shape[:2]
        pe = self.pe.unsqueeze(0)[:,:seqLen,:]
        pe = pe.repeat(batchSize, 1, 1).to(emb.device)
        return torch.cat((emb, pe), dim=2)

class InputEmbedding(nn.Module):
    def __init__(self, nCategories, embSize, peSize, dropout):
        super().__init__()
        self.pe = PositionalEncoding(peSize)
        self.em = nn.Embedding(nCategories, embSize - 1)

    def forward(self, categoryIdx, vals):
        emb = self.em(categoryIdx)
        vals = vals.unsqueeze(-1)
        emb = torch.cat((emb, vals), dim=-1)

        emb = self.pe(emb)
        print(emb.shape)
        exit(0)
    

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

        return emb

class MimicModel(nn.Module):
    def __init__(self, nLayers, embSize, nHeads=8, pwffDim=512, dropout=0.1):
        super().__init__()
        self.nLayers = nLayers
        self.embSize = embSize
        self.blocks = nn.ModuleList([Block(embSize, nHeads, pwffDim, dropout)
                                     for n in range(nLayers)])

    def forward(self, emb, mask=None):
        for layer in self.blocks:
            emb = layer(emb, mask)

        return emb

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batchSize = 16
    #hrs per day, number of days, events per hour
    #1440
    seqLen = 24 * 30 * 2 
    nLayers = 4
    nTokens = 2350
    embSize = 112
    peSize = 16
    dropout = 0.1

    x = torch.randint(0, nCategories, size=(batchSize, seqLen)).to(device)
    vals = torch.randn(batchSize, seqLen).to(device)


    createEmb = InputEmbedding(nCategories, embSize, peSize, dropout).to(device)
    inputEmb = createEmb(x, vals)

    model = MimicModel(nLayers, embSize + peSize).to(device)

    y = model(inputEmb)
    print(y.shape)
    return

if __name__ == '__main__':
    main()