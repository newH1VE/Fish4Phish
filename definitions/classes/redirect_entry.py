# class for redirect entry

class RedirectEntry:
    def __init__(self, url, redirects, protocol, content):
        self.url = url
        self.redirects = redirects
        self.protocol = protocol
        self.content = content
