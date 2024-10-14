# JazzBot
Project by M. Basse, C. Soto, H. Duprieu, L. Stöckel, P. Aguié, conducted as part of the 2nd year curriculum, supervised by Jean-Pierre Briot (CNRS) and Gaëtan Hadjeres (Sony AI).

This repository contains the code used to build and train JazzBot, a transformer model used for jazz solo generation in MIDI format. The model's aim is to generate sequences of notes following a starting sequence given as an input.

The solos are written in the MIDI format. This allows us to tokenize music pieces by writing solos as a sequence of notes, from which we extact four attributes: pitch, duration, time shift (i.e. the duration between the start the previous note and the start of the current note), and velocity. Each possible value of these attribute is a token in our custom vocabulary.


1. `model.py` defines the JazzBot module, using Torch's nn.Transformer and nn.Embedding, and the PositionalEncoding module, used in the JazzBot module.

2. `data_processor.py` defines the functions required to process datasets of MIDI files and convert them into sequences of tokens. It also contains functions that can directly process .csv files containing sequences of tokens.

3. `data_loader.py` defines functions that creates datasets and data loaders.

4. `vocab.py` defines the custom vocabulary created for the music generation task. Each note is decomposed into 4 attributes (pitch, duration, time shift, velocity), and we define tokens for each possible value of each attribute.

5. `train.py` and `generate.py` contain the functions used to respectively train our models, and generate music using trained models.