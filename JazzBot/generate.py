from JazzBot.config import *
from JazzBot.model import JazzBot
from JazzBot.data_processor import *


def generate(model: JazzBot, start_sequence, N, max_len=1000):
    '''
    Generate a sequence of notes using the model
    Args:
    - model : the trained model
    - start_sequence : the sequence of notes to start the generation with
    - N : the number of notes in each piece
    - max_len : the maximum length of the generated sequence
    Returns:
    - the generated sequence of notes
    '''
    generated_sequence = start_sequence
    for i in range(max_len):
        input_sequence = torch.tensor(pieceToInputTarget(tokensToPieces(generated_sequence,N)[0])[0]).to(device)
        tgt_mask = model.get_tgt_mask(input_sequence[0].size(0)).to(device)
        pred = model(torch.tensor([0]*len(N+4)).to(device), input_sequence, tgt_mask)
        next_item = pred.topk(1)[1].view(-1)[-1].item()
        generated_sequence += [next_item]
        if next_item == 3:
            break
    return generated_sequence