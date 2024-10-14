# JazzBot
Project by M. Basse, C. Soto, H. Duprieu, L. Stöckel, P. Aguié, conducted as part of the 2nd year curriculum, supervised by Jean-Pierre Briot (CNRS) and Gaëtan Hadjeres (Sony AI).

This repository contains the code used to build and train JazzBot, a transformer model used for jazz solo generation in MIDI format. The solos are written in the MIDI format. This allows us to tokenize music pieces by writing solos as a sequence of notes, from which we extact four attributes: pitch, duration, time shift (i.e. the duration between the start the previous note and the start of the current note), and velocity. Each possible value of these attribute is a token in our custom vocabulary.


1. `model.py` contains the implementation of the JazzBot transformer model. It includes the architecture, layers, and functions necessary for training and generating jazz solos.

2. `data_processing.py`: This file handles the data preprocessing for the JazzBot model. It includes functions for tokenizing MIDI files, extracting attributes, and creating the custom vocabulary.

3. `train.py`: This file is responsible for training the JazzBot model. It includes functions for loading and preprocessing the training data, defining the training loop, and saving the trained model.

4. `generate.py`: This file allows you to generate jazz solos using the trained JazzBot model. It includes functions for loading the trained model, setting the generation parameters, and generating MIDI files.