import torch

from model import YelpModel, createMask
from yelp_dataset import YelpDataset

def main():

    validset = YelpDataset('valid')

    device = 'cpu'
    vocabSize = len(validset.tokToIdx.keys())
    seqLen = validset.maxLen
    nLayers = 4
    embSize = 256
    padIdx = validset.tokToIdx['<pad>']
    startIdx = validset.tokToIdx['<start>']
    endIdx = validset.tokToIdx['<end>']

    model = YelpModel(nLayers, embSize, vocabSize, device)
    model.load_state_dict(torch.load('model.pt'))
    model = model.to(device)
    model.eval()

    with torch.no_grad():

        out = torch.tensor(startIdx).view(1, 1).long()
        for i in range(seqLen):

            mask = createMask(out, padIdx)

            pred = model(out, mask)

            #randomly pick some of the highest ones
            print(pred.shape)
            exit(0)
            predTok = pred.argmax(2)[:,-1].view(1, 1).long()

            out = torch.cat([out, predTok], dim=1)

            if predTok[0, 0].item() == endIdx:
                break
                
        out = out.squeeze(0)
        predSentence = ' '.join([validset.idxToTok[int(idx)] for idx in out])
        print(predSentence)
        return

    return
    with torch.no_grad():
        for item in validset:
            tokTen = item['tokTen'].to(device)
            tokTen = tokTen.unsqueeze(0)

            tokInput = tokTen[:,:-1]

            mask = createMask(tokInput, padIdx).to(device)

            pred = model(tokInput, mask)

            predIdx = torch.argmax(pred, dim=2).squeeze(0)
            tokTrue = tokTen[:,1:].squeeze(0)

            predSentence = ' '.join([validset.idxToTok[int(idx)] for idx in predIdx])
            trueSentence = ' '.join([validset.idxToTok[int(idx)] for idx in tokTrue])
            print(predSentence)
            print(trueSentence)

            return


if __name__ == '__main__':
    main()