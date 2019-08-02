class Word:
    def __init__(self, word_string, count=1, w_type=None):
        self.prototype = word_string
        self.count = count
        self.w_type = w_type

    def __add__(self, another):
        if self.prototype == another.prototype and self.w_type == another.w_type:
            self.count += another.count
            return self
        raise Exception("Only the same words of the same type can be added together")

    def __repr__(self):
        return "{}\t{}\t{}".format(self.prototype, self.count, self.w_type)

    def __hash__(self):
        return hash((self.prototype, self.count, self.w_type))