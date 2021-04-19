


class RedirectEntry:

    def __init__(self, url, redirects, protocol, content):
        """
        url: url checked for redirects
        redirects: number of redirects
        protocol: used protocol (1: HTTPS, 0: HTTP)
        content: content of website
        """
        self.url = url
        self.redirects = redirects
        self.protocol = protocol
        self.content = content
