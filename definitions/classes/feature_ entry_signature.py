

"""
 class for signature entry containing the url, most frequent terms and entities
 as well as certificate subject
"""

class SignatureEntry:
    def __init__(self, url, label, final_url, cert_subject, ent1, ent2, ent3, ent4,ent5, term1, term2, term3, term4,term5):
        self.url = url
        self.label = label
        self.final_url = final_url
        self.cert_subject = cert_subject
        self.ent1 = ent1
        self.ent2 = ent2
        self.ent3 = ent3
        self.ent4 = ent4
        self.ent5 = ent5
        self.term1 = term1
        self.term2 = term2
        self.term3 = term3
        self.term4 = term4
        self.term5 = term5


    def get_ents(self):
        ents = []
        ents.append(self.ent1)
        ents.append(self.ent2)
        ents.append(self.ent3)
        ents.append(self.ent4)
        ents.append(self.ent5)

        return ents

    def get_terms(self):
        terms = []
        terms.append(self.term1)
        terms.append(self.term2)
        terms.append(self.term3)
        terms.append(self.term4)
        terms.append(self.term5)

        return terms



