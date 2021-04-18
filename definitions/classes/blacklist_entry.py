# class for redirect entry

class BlacklistEntry:
    def __init__(self, domainname, not_after):
        self.domainname = domainname
        self.not_after = not_after
