import os
import music21 as m21
import json
import tensorflow.keras as keras
import numpy as np

KERN_DATASET_PATH = "Data_preprocessing\\deutschl\\erk"
SAVE_DIR = "Data_preprocessing\\Dataset"
SINGLE_DATASET = "Data_preprocessing\\single_file_dataset"
SEQUENCE_LENGTH = 64
MAPPING_PATH = "Data_preprocessing\\mapping.json"
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]
def load_songs_in_kern(dataset_path):
    songs = []

    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == 'krn':
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True
def load(file_path):
    with open(file_path, 'r') as file:
        song = file.read()
    return song
    
def transpose(song):
    
    # get the key
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    # print(key)
    # estimate the key
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    
    # get interval for transposition eg: Bmaj -> Cmaj
    
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    
    # transpose the song
    transposed_song = song.transpose(interval)
    
    return transposed_song
def encode_song(song, time_step=0.25):
    # pitch = 60, duration = 1 -> [60, "_", "_", "_"]
    encoded_song = []
    for event in song.flat.notesAndRests:
        
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        
        # convert the note/rest into time series notation
        
        steps = int(event.duration.quarterLength/time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")   
    
    # cast the string
    encoded_song = " ".join(map(str, encoded_song))  
    return encoded_song
def preprocess(dataset_path):
    
    ## load songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):
        
        # filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        
        # transpose the song to Cmaj/Amin
        song = transpose(song)
        
        # encode song in time series representation
        encoded_song = encode_song(song)
        
        # save song to the text file 
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as file:
            file.write(encoded_song)
def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    
    new_song_delimiter = "/ "*sequence_length
    
    songs = "";
    
    for path, _, files in os.walk(dataset_path):
         for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    
    songs = songs[:-1]
    
    # save the the songs
    with open(file_dataset_path, "w") as file:
        file.write(songs)
    return songs 
def create_mapping(songs, mapping_path):
    mapping = {}
    
    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    
    # create the mapping
    for i, symbol in enumerate(vocabulary):
        mapping[symbol] = i
    
    # save vocabulary to json
    with open(MAPPING_PATH, "w") as file:
        json.dump(mapping, file, indent=4)

def map_the_songs(songs):
    int_songs = []
    mapping = {}
    
    # load json file
    with open(MAPPING_PATH, "r") as file:
        mapping = json.load(file)
    
    # map songs to int
    for symbol in songs.split():
        int_songs.append(mapping[symbol])
    
    return int_songs

def generate_training_sequences(sequence_length):
    # [11, 12, 13, 14, .....] -> input: [11, 12], target: [13], input: [12, 13], target: [14]
    
    # load the song and map them to int
    songs = load(SINGLE_DATASET)
    int_songs = map_the_songs(songs)
    
    num_seq = len(int_songs)-sequence_length
    input_sequences = []
    target = []
    for i in range(num_seq):
        input_sequences.append(int_songs[i:i+sequence_length])
        target.append(int_songs[i+sequence_length])
    # one hot encoding
    # input:(num_seq, length_sequence) -> onehot encoding -> input(num_seq, length_sequence, vocabulary_length)
    # [[1, 2, 0], [1, 1, 2]] -> [[[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, 1, 0][0, 1, 0][0, 0, 1]]]
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(input_sequences, num_classes=vocabulary_size)
    targets = np.array(target)
    
    return inputs, targets
def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    # print(inputs.shape, targets.shape)
    # print(targets)
    a = 1
if __name__ == "__main__":
    
    main()