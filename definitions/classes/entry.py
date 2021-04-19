
# standard entry containing a url and coresponding label

class Entry:
    def __init__(self, label, url):
        self.url = url
        self.label = label

    def __eq__(self, other):
        if self.url == other.url and self.label == other.label:
            return True
        else:
            return False