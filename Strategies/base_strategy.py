class Strategy:
    def __init__(self, name="base"):
        self.name = name

    def generate_signals(self, panel, breadth):
        raise NotImplementedError
