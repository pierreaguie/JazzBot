from JazzBot.vocab import *
from music21 import *
import os

def noteToToken(n,l):
    '''
    Converts the note n to tokens (note, duration, time shift, velocity), using the time of the last note l to compute the time shift
    '''
    note_pitch = n.pitch.midi
    last_time = l
    note_duration = int(n.duration.quarterLength*4)
    note_offset = int(n.offset*4 - last_time)
    last_time = n.offset*4
    note_velocity = n.volume.velocity

    return [NOTE_TOKS[note_pitch],
            DUR_TOKS[note_duration],
            TIM_TOKS[note_offset],
            VEL_TOKS[note_velocity]]

def midiToTokens(folder_path,f):
    '''
    Reads a MIDI file f located in folder_path, and yields the sequence of tokens corresponding to the notes in the file
    (does not add the start of sequence and end of sequence tokens)
    '''
    tokens = []
    midi_file = midi.MidiFile()
    midi_file.open(folder_path +f)
    midi_file.read()
    midi_file.close()
    stream = midi.translate.midiFileToStream(midi_file)
    # Iterate over the notes in the stream and extract the note information
    last_time = 0
    for note in stream.flat.notes:
        if note.isNote:
            tokens+=noteToToken(note,last_time)

        if note.isChord:
            for n in note:
                tokens+=noteToToken(n,last_time)

    return tokens

def tokensToPieces(t,N):
    '''
    parameters : t = tokens (4*(nb_tok))  
    cut the list of tokens into pieces of length N, add PAD if necessary
    '''
    pieces = []
    nb_tok = (len(t))//4
    for i in range(nb_tok-N+1):
        pieces.append(t[4*i:4*i+N])
    return pieces

def pieceToInputTarget(p):
    '''
    return a couple of vectors (input, target) with input = SOS + piece; target = piece + EOS
    '''
    input_p = cv_sos()
    target_p = []
    for i in range(len(p)//4):
        input_p+= custom_vocab(p[4*i:4*(i+1)])
        target_p += custom_vocab(p[4*i:4*(i+1)])
    target_p += cv_eos()
    return(input_p,target_p)

def folderToVectInputTarget(folder_path, N):
    '''
    parameters : folder_path, N = number of notes in pieces 
    '''
    vectInput = []
    vectTarget = []
    file_names = os.listdir(folder_path)
    for f in file_names:
        tokens = midiToTokens(folder_path,f)
        pieces = tokensToPieces(tokens,4*N)
        for p in pieces:
            input,target = pieceToInputTarget(p)
            vectInput.append(input)
            vectTarget.append(target)
    return vectInput,vectTarget

def csvToVectInputTarget(file_path, N):
    '''
    parameters : file_path, N = number of notes in pieces 
    '''
    vectInput = []
    vectTarget = []
    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.split(',')
            pieces = tokensToPieces(tokens, N)
            for p in pieces:
                input,target = pieceToInputTarget(p)
                vectInput.append(input)
                vectTarget.append(target)
    return vectInput,vectTarget