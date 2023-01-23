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
    def __init__(self, embSize, dropout, device, maxLen=1000):
        super().__init__()
        maxLen = 1000
        denominator = torch.exp(torch.arange(0, embSize, 2) * torch.log(torch.tensor(10000)) / embSize)
        pos = torch.arange(0, maxLen).reshape(maxLen, 1)

        pe = torch.zeros((maxLen, embSize), dtype=torch.float)
        pe[:, 0::2] = torch.sin(pos / denominator)
        pe[:, 1::2] = torch.cos(pos / denominator)
        pe.requires_grad = False

        import matplotlib.pyplot as plt
        plt.plot(pe[:500, :])
        plt.show()
        exit(0)

        self.pe = pe.unsqueeze(0).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding):
        return self.dropout(embedding + self.pe[:,:embedding.shape[1],:])

class PositionWiseFeedforward(nn.Module):
    def __init__(self, embSize, pwffDim):
        super().__init__()
        self.fc1 = nn.Linear(embSize, pwffDim)
        self.fc2 = nn.Linear(pwffDim, embSize)
        self.relu = nn.ReLU() 

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

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

        #linearly project to dq, dk, and dv
        #go from [batchSize, seqLen, embSize]
        #to      [batchSize, seqLen, nHeads, headDim]
        #then permute
        #        [batchSize, nHeads, seqLen, headDim]
        Q = Q.view(batchSize, -1, self.nHeads, self.headDim).permute(0, 2, 1, 3)
        K = K.view(batchSize, -1, self.nHeads, self.headDim).permute(0, 2, 1, 3)
        V = V.view(batchSize, -1, self.nHeads, self.headDim).permute(0, 2, 1, 3)

        #you want to matmul Q and K
        #where Q is [batchSize, nHeads, seqLen, headDim]
        #and K is   [batchSize, nHeads, headDim, seqLen]
        #so the output is [batchSize, nHeads, seqLen, seqLen]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        #softmax along the last dimension
        attention = torch.softmax(energy, dim=-1)

        #dropout applied to attention in tensorflow version?
        out = torch.matmul(attention, V)
        #attention[seqLen, seqLen] X V[seqLen, headDim]
        #out[seqLen, headDim]

        #put the indices in right place [batchSize, seqLen, nHeads, headDim]
        out = out.permute(0, 2, 1, 3).contiguous()
        #reshape it so its batch to [batchSize, seqLen, embSize]
        out = out.view(batchSize, -1, self.embSize)

        out = self.fcOut(out)

        return out, attention

#nDecoderLayers, embSize, nHeads, pwffDim, dropout)
class Decoder(nn.Module):
    def __init__(self, nLayers, embSize, nHeads, pwffDim, dropout):
        super().__init__()
        self.nLayers = nLayers
        self.embSize = embSize
        self.decoderLayers = nn.ModuleList([DecoderLayer(embSize, nHeads, pwffDim, dropout)
                                            for n in range(nLayers)])

    def forward(self, encOut, tarEmb, tarMask):
        for layer in self.decoderLayers:
            tarEmb, attention = layer(encOut, tarEmb, tarMask)

        return tarEmb, attention

class DecoderLayer(nn.Module):
    '''
    consists of 3 sublayers
    first: masked multihead attention
    second: multihead attention over output of encoder stack
    third: positionwise fully connected feed forward
    residual connection around each sublayers 
    followed by layer normalization
    dropout is applied to output of each sublayer 
    '''
    def __init__(self, embSize, nHeads, pwffDim, dropout):
        super().__init__()
        self.maskedSelfAttention = MultiHeadAttention(embSize, nHeads)
        self.maskedAttentionLayerNorm = nn.LayerNorm(embSize)

        self.encAttention = MultiHeadAttention(embSize, nHeads)
        self.encAttentionLayerNorm = nn.LayerNorm(embSize)

        self.pwff = PositionWiseFeedforward(embSize, pwffDim)
        self.pwffLayerNorm = nn.LayerNorm(embSize)

        self.dropout = nn.Dropout(dropout)

    def forward(self, encOut, tarEmb, tarMask):
        #first sublayer masked self attention
        out, maskedAttention = self.maskedSelfAttention(tarEmb, tarEmb, tarEmb, tarMask)
        #residual connection and layer norm
        tarEmb = self.maskedAttentionLayerNorm(tarEmb + self.dropout(out))

        #second sublayer
        #query, key, value
        #of the encoder output, what should we attend to
        out, attention = self.encAttention(tarEmb, encOut, encOut)
        tarEmb = self.encAttentionLayerNorm(tarEmb + self.dropout(out))

        #third sublayer
        out = self.pwff(tarEmb)
        tarEmb = self.pwffLayerNorm(tarEmb + self.dropout(out))

        return tarEmb, attention

class Encoder(nn.Module):
    def __init__(self, nLayers, embSize, nHeads, pwffDim, dropout):
        super().__init__()
        self.nLayers = nLayers
        self.embSize = embSize
        self.encoderLayers = nn.ModuleList([EncoderLayer(embSize, nHeads, pwffDim, dropout)
                                            for n in range(nLayers)])
    def forward(self, emb):

        for layer in self.encoderLayers:
            emb, attention = layer(emb)

        return emb


class EncoderLayer(nn.Module):
    '''
    consists of 2 sublayers
    first: multihead self attention
    second: positionwise fully connected feed-forward
    residual connection around each two sublayers 
    followed by layer normalization
    dropout is applied to output of each sublayer 
    '''
    def __init__(self, embSize, nHeads, pwffDim, dropout):
        super().__init__()
        self.selfAttention = MultiHeadAttention(embSize, nHeads)
        self.attentionLayerNorm = nn.LayerNorm(embSize)

        self.pwff = PositionWiseFeedforward(embSize, pwffDim)
        self.pwffLayerNorm = nn.LayerNorm(embSize)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, emb):
        #first sublayer
        out, attention = self.selfAttention(emb, emb, emb)
        #residual followed by layerNorm
        emb = self.attentionLayerNorm(emb + self.dropout(out))

        #second sublayer
        out = self.pwff(emb)
        emb = self.pwffLayerNorm(emb + self.dropout(out))
        return emb, attention

class Transformer(nn.Module):
    def __init__(self, nEncoderLayers, nDecoderLayers, embSize, nHeads,
                 srcVocabSize, tarVocabSize, device, pwffDim=2048, dropout=0.1):
        super().__init__()
        #embeddings for source vocab
        self.srcEmb = nn.Embedding(srcVocabSize, embSize)
        #embeddings for target vocab
        self.tarEmb = nn.Embedding(tarVocabSize, embSize)
        #positional encoding
        self.pe = PositionalEncoding(embSize, dropout, device)

        #encoder
        self.encoder = Encoder(nEncoderLayers, embSize, nHeads, pwffDim, dropout)
        
        #decoder
        self.decoder = Decoder(nDecoderLayers, embSize, nHeads, pwffDim, dropout)

        #output
        self.fcOut = nn.Linear(embSize, tarVocabSize)


    def forward(self, src, tar, tarMask):

        encOut = self.encode(src)
        decOut, attention = self.decode(encOut, tar, tarMask)

        return self.fcOut(decOut)

    def encode(self, src):
        srcEmb = self.pe(self.srcEmb(src))
        encOut = self.encoder(srcEmb)

        return encOut

    def decode(self, encOut, tar, tarMask):
        tarEmb = self.pe(self.tarEmb(tar))
        decOut, attention = self.decoder(encOut, tarEmb, tarMask)

        return decOut, attention


def createForwardMask(seqLen):
    '''
    Prevents model from attending to tensors its supposed to predict
    Remember
    softmax(z) = e^z / sum(e^z), so if z is a large negative number
    then e^z is really close to zero
    '''
    mask = torch.tril(torch.ones((seqLen, seqLen)))

    return mask

def createPaddingMask(seq, padIdx):

    padMask = torch.ones(seq.shape)
    padMask[seq == padIdx] = 0

    return padMask

def main():

    batchSize = 32
    nEncoderLayers = 3
    nDecoderLayers = 3
    nHeads = 8
    srcVocabSize = 10000
    tarVocabSize = 5000
    embSize = 512
    srcSeqLen = 10
    tarSeqLen = 7
    device = 'cuda'

    src = torch.randint(0, srcVocabSize, size=(batchSize, srcSeqLen))
    tar = torch.randint(0, tarVocabSize, size=(batchSize, tarSeqLen))

    tarMask = createForwardMask(tarSeqLen)

    tarMask = tarMask.to(device)
    src = src.to(device)
    tar = tar.to(device)

    model = Transformer(nEncoderLayers, nDecoderLayers, embSize, nHeads,
                        srcVocabSize, tarVocabSize, device)

    model = model.to(device)

    pred = model(src, tar, tarMask)
    print(pred.shape)

if __name__ == '__main__':
    main()