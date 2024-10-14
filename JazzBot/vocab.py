from torchtext.vocab import Vocab
from collections import OrderedDict


# Special tokens
SOS = "sos" # Start of sequence
EOS = "eos" # End of sequence
PAD = "xxx"  # Padding token

SPECIAL = [SOS,EOS,PAD]

# Number of tokens for each type: pitch, duration, time shift, velocity
NOTE_SIZE = 128
DUR_SIZE = 160
TIM_SIZE = 1000
VEL_SIZE = 128

# Tokens for each type
NOTE_TOKS = [f'n{i}' for i in range(NOTE_SIZE)] 
DUR_TOKS = [f'd{i}' for i in range(DUR_SIZE)]
TIM_TOKS = [f't{i}' for i in range(TIM_SIZE)]
VEL_TOKS = [f'v{i}' for i in range(VEL_SIZE)]

# The dummy token is only used so that the tokens index start at 1, as required by the vocab() function
VOCAB = ["dummy"] + SPECIAL + NOTE_TOKS + DUR_TOKS + SPECIAL + TIM_TOKS + VEL_TOKS 
DICT = [(element, index) for index, element in enumerate(VOCAB)]

# Custom vocabulary
CV = Vocab(OrderedDict(DICT), specials=SPECIAL, min_freq=1)

itos_vocab = CV.itos
vocab_size = len(itos_vocab)

def custom_vocab(tok):
    '''
    tok = [note,dur,tim,vel]
    '''
    return [CV[tok[0]],
            CV[tok[1]],
            CV[tok[2]],
            CV[tok[3]]]

def cv_eos():
    '''
    return token corresponding to eos
    '''
    return [CV[EOS]]

def cv_sos():
    '''
    return token corresponding to sos
    '''
    return [CV[SOS]]