import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import music21 as m21
import os

from JazzBot.config import *
from JazzBot.model import *
from JazzBot.data_processor import *
from JazzBot.vocab import *
from JazzBot.data_decoder import *
from JazzBot.generate import *

st.title("JazzBot")
st.text("Use JazzBot to generate jazz solos!")

st.text("Instructions : Drag the beginning of your solo as a .midi file below, \nchoose the maximum length of your piece and the tempo, and click on \"Generate\"")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load("Models/model4out.pth",map_location=torch.device(device))
model = Transformer(num_tokens=len(custom_vocab), dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1).to(device)
model.load_state_dict(state_dict)
model.eval()

file = st.file_uploader("Upload a MIDI file",type=['mid'])


bpm = st.slider("BPM",min_value=60,max_value=200,value=120)
maxl = st.slider("Maximum length of the generated sequence",min_value=20,max_value=400,value=100)

if file is not None:
    with open(os.path.join("midi_input/",file.name),"wb") as f:
        f.write(file.getbuffer())
    clicked = st.button('Generate')

    if clicked:
        start_tokens = [custom_vocab[elt] for elt in midiToTokens("midi_input/",file.name)]
        generated_tokens = generate_sequence(model, start_tokens, max_length=maxl, temperature=1.0)
        decoded_tokens = [itos_vocab[el] for el in generated_tokens]
        tokens_to_midi(decoded_tokens, "midi_output/" + file.name, bpm)
        with open(os.path.join("midi_output/",file.name),"rb") as f:
            downloaded = st.download_button("Download",data = f, file_name = "[JazzBoted]" + file.name)