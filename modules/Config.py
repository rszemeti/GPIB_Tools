import json

class Config:
    def __init__(self):
        self.analyserAddress = None
        self.audioSource = None
        self.sample_rate = None
        self.start_freq=1E3
        self.end_freq=10E6
        self.mode='RF'
        self.changeover=20E3
        

    def save_to_file(self, filename = '.config_phase_noise.json'):
        data = {
            'analyserAddress': self.analyserAddress,
            'audioSource': self.audioSource,
            'sample_rate': self.sample_rate,
            'start_freq': self.start_freq,
            'end_freq': self.end_freq,
            'mode': self.mode,
            'changeover': self.changeover
        }
        with open(filename, 'w') as file:
            json.dump(data, file)

    @staticmethod
    def load_from_file(filename='.config_phase_noise.json'):
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                config = Config()
                config.analyserAddress = data.get('analyserAddress')
                config.audioSource = data.get('audioSource')
                config.sample_rate = data.get('sample_rate',48000)
                config.start_freq = data.get('start_freq',1E3)
                config.end_freq = data.get('end_freq',10E6)
                config.mode = data.get('mode',"RF")
                config.changeover = data.get('changeover',20E3)
                return config
        except FileNotFoundError:
            return Config()
