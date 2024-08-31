import tensorflow.keras as keras
import json
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:
    
    def __init__(self, model_path="Generating_Melody\\model.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        
        with open(MAPPING_PATH, 'r') as file:
            self.mapping = json.load(file)
        
        self._start_symbols = ["/"]*SEQUENCE_LENGTH
        
    
    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        # seed -> "64 _ 65 _ _"  
        
        seed = seed.split()
        melody = seed
        seed - self._start_symbols + seed
        
        # mapping seed to integers
        
        seed_ints = [self.mapping[symbol] for symbol in seed]
        
        for _ in range(num_steps):
            