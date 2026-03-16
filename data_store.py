from embeddings import generate_embedding

class DataStore:

    def __init__(self):
        self.texts = []
        self.embeddings = []

    def add_text(self, text):

        emb = generate_embedding(text)

        self.texts.append(text)
        self.embeddings.append(emb)

    def get_all(self):

        return self.texts, self.embeddings