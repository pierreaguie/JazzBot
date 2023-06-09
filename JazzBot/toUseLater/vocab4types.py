from torchtext.vocab import vocab
from collections import OrderedDict

#special tokens
SOS = "sos" #start
EOS = "eos" #end
PAD = "xxx"  #NAN

SPECIAL = [SOS,EOS,PAD]

NOTE_SIZE = 128
DUR_SIZE = 160
TIM_SIZE = 1000
VEL_SIZE = 128

NOTE_TOKS = [f'n{i}' for i in range(NOTE_SIZE)] 
DUR_TOKS = [f'd{i}' for i in range(DUR_SIZE)]
TIM_TOKS = [f't{i}' for i in range(TIM_SIZE)]
VEL_TOKS = [f'v{i}' for i in range(VEL_SIZE)]

# Le token dummy sert seulement à initialiser les mots du vocab à partir de l'index 1, conformément aux prérequis de la fonction vocab()
#TODO mettre le SPECIAL avant
VOCAB_NOTE = ["dummy"] + NOTE_TOKS + SPECIAL 
VOCAB_DUR = ["dummy"] + DUR_TOKS + SPECIAL
VOCAB_TIM = ["dummy"] + TIM_TOKS + SPECIAL
VOCAB_VEL = ["dummy"] + VEL_TOKS + SPECIAL

DICT_NOTE = [(element, index) for index, element in enumerate(VOCAB_NOTE)]
DICT_DUR = [(element, index) for index, element in enumerate(VOCAB_DUR)]
DICT_TIM = [(element, index) for index, element in enumerate(VOCAB_TIM)]
DICT_VEL = [(element, index) for index, element in enumerate(VOCAB_VEL)]

CV_NOTE = vocab(OrderedDict(DICT_NOTE))
CV_DUR = vocab(OrderedDict(DICT_DUR))
CV_TIM = vocab(OrderedDict(DICT_TIM))
CV_VEL = vocab(OrderedDict(DICT_VEL))

itos_NOTE = CV_NOTE.get_itos()
itos_DUR = CV_DUR.get_itos()
itos_TIM = CV_TIM.get_itos()
itos_VEL = CV_VEL.get_itos()


def custom_vocab(tok):
    '''
    tok = [note,dur,tim,vel]
    '''
    return [CV_NOTE[tok[0]],
            CV_DUR[tok[1]],
            CV_TIM[tok[2]],
            CV_VEL[tok[3]]]

def cv_eos():
    '''
    return token corresponding to eos
    '''
    return [CV_NOTE[EOS],
            CV_DUR[EOS],
            CV_TIM[EOS],
            CV_VEL[EOS]]

def cv_sos():
    '''
    return token corresponding to sos
    '''
    return [CV_NOTE[SOS],
            CV_DUR[SOS],
            CV_TIM[SOS],
            CV_VEL[SOS]]