import torch

from model import MimicModel, createMask
from dataset import MimicDataset

def main():

    validset = MimicDataset('valid.txt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nTokens = len(validset.tokToIdx.keys())
    nLayers = 4
    embSize = 256
    peSize = 64
    seqLen = validset.seqLen
    maxHrs = validset.maxHrs

    padIdx = validset.tokToIdx['<pad>']
    clsIdx = validset.tokToIdx['<cls>']
    endIdx = validset.tokToIdx['<end>']

    model = MimicModel(nLayers, embSize, nTokens, peSize, device, maxHrs=maxHrs)
    model.load_state_dict(torch.load('model.pt'))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        #start off with demographics and cls Idx

        tokOut = torch.tensor(clsIdx).view(1,1).long()
        valsOut = torch.tensor(0).view(1,1).float()
        timesOut = torch.tensor(0).view(1, 1).long()

        tokOut = tokOut.to(device)
        valsOut = valsOut.to(device)
        timesOut = timesOut.to(device)

        for i in range(seqLen):
            
            mask = createMask(tokOut, padIdx).to(device)

            out = model(tokOut, valsOut, timesOut, mask)
            tokPred, valsPred, timesPred = model.unsupervised(out)

            tokPred = tokPred.argmax(2)[:,-1].view(1, 1).long()
            valsPred = valsPred.squeeze(-1)[:,-1].view(1, 1).float()
            timesPred = timesPred.argmax(2)[:,-1].view(1, 1).long()

            tokOut = torch.cat([tokOut, tokPred], dim=-1)
            valsOut = torch.cat([valsOut, valsPred], dim=-1)
            timesOut = torch.cat([timesOut, timesPred], dim=-1)

            if tokPred[0, 0].item() == endIdx:
                break
        
        tokIdx = tokOut.cpu().numpy()[0,:100]
        timesIdx = timesOut.cpu().numpy()[0,:100]

        tok = [validset.idxToTok[i] for i in tokIdx]
        
        print(tok)
        print(timesIdx)



if __name__ == '__main__':
    main()