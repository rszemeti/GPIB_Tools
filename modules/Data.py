from typing import List

class Trace:
    def __init__(self, bw, points=None):
        self.bw = bw
        self.points = points if points is not None else []

    def addPoint(self, point):
        self.points.append(point)

    def getPoints(self):
        return self.points

class ResultSet:
    def __init__(self,startF,endF):
        self.startF=startF
        self.endF=endF
        self.traces=[]

    def addTrace(self, trace: Trace):
        self.traces.append(trace)

    def getTraces(self) ->List[Trace]:
        return self.traces

    def setFreq(self,freq):
        self.freq=freq

    def setLevel(self,lvl):
        self.level=lvl

    def save(self, filename):
        data = {
            'startF': self.startF,
            'endF': self.endF,
            'freq': self.freq,
            'level': self.level,
            'traces': [trace.__dict__ for trace in self.traces]
        }
        with open(filename, 'w') as file:
            json.dump(data, file)

    def load(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            self.startF = data['startF']
            self.endF = data['endF']
            try:
                self.freq = data['freq']
                self.level = data['level']
            except KeyError as e:
                print(f"Missing key in JSON data: {e}")
                self.freq = 1
                self.level = 0
                            
            self.traces = [Trace(**trace) for trace in data['traces']]
