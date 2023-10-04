import json

class Config:
    def __init__(self):
        self.analyserAddress = None
        self.audioSource = None
        self.sample_rate = None

    def save_to_file(self, filename):
        data = {
            'analyserAddress': self.analyserAddress,
            'audioSource': self.audioSource,
            'sample_rate': self.sample_rate
        }
        with open(filename, 'w') as file:
            json.dump(data, file)

    @staticmethod
    def load_from_file(filename):
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                config = Config()
                config.analyserAddress = data.get('analyserAddress')
                config.audioSource = data.get('audioSource')
                config.sample_rate = data.get('sample_rate')
                return config
        except FileNotFoundError:
            return Config()
