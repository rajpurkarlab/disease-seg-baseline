import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class ContrastiveCorpusSimilarity():
    """
    Computes the contrastive corpus similarity of an explicand, relative to a given corpus and foil set.
    """
    def __init__(self, encoder, explicand_encoder, corpus_tokens, foil_tokens):
        """
        Constructor for `ContrastiveCorpusSimilarity`.

        Parameters:
            encoder (torch.nn.Module): Text encoder for encoding corpus and foil captions
            explicand_encoder (torch.nn.Module): Image encoder for encoding explicand
            corpus_dataloader (torch.utils.data.DataLoader): PyTorch DataLoader containing all tokenized corpus captions
            # foil_dataloader (torch.utils.data.DataLoader): PyTorch DataLoader containing all tokenized foil captions
        """
        self.encoder = encoder
        self.explicand_encoder = explicand_encoder             
        self.corpus_tokens = corpus_tokens
        self.foil_tokens = foil_tokens

        # Obtain encodings and sizes of corpus and foil data
        self.corpus_encodings, self.foil_encodings, self.corpus_size, self.foil_size = self.get_corpus_foil_encodings(corpus_tokens, foil_tokens)

    def get_corpus_foil_encodings(self, corpus_dataloader, foil_tokens):
        """
        Returns encodings and sizes of corpus and foil data.

        Parameters:
            corpus_dataloader (torch.utils.data.DataLoader): PyTorch DataLoader containing tokenized (but not encoded) corpus captions
            # foil_dataloader (torch.utils.data.DataLoader): PyTorch DataLoader containing tokenized (but not encoded) foil captions

        Returns:
            Tuple containing:
            - corpus_encodings_dataloader (torch.utils.data.DataLoader): PyTorch DataLoader containing encoded corpus captions
            - foil_encodings_dataloader (torch.utils.data.DataLoader): PyTorch DataLoader containing encoded foil captions
            - corpus_encodings size (int): number of encoded corpus captions
            - foil_encodings size (int): number of encoded foil captions
        """

        # Encode corpus and foil data by passing them through the text encoder `self.encoder`   
        corpus_encodings, foil_encodings = self.encode_data(corpus_dataloader), self.encode_data(foil_tokens)
        
        # Convert corpus and foil encodings to PyTorch DataLoaders
        corpus_encodings_dataloader = DataLoader(TensorDataset(corpus_encodings), batch_size=64, shuffle=False)
        foil_encodings_dataloader = DataLoader(TensorDataset(foil_encodings), batch_size=64, shuffle=False)
        
        return corpus_encodings_dataloader, foil_encodings_dataloader, corpus_encodings.size(0), foil_encodings.size(0)

    def compute_contrastive_corpus_similarity(self, explicand):
        """
        Computes the contrastive corpus similarity of the explicand, relative to the encoded corpus and foil captions.
        Calculation follows Equation (6), p. 8 of COCOA paper.

        Parameters:
            explicand (torch.Tensor): input image for which to compute contrastive corpus similarity

        Returns:
            Contrastive corpus similarity (float)
        """
        corpus_similarity = self.compute_representation_similarity(explicand, self.corpus_encodings, self.corpus_size) # Compute similarity between explicand and corpus
        foil_similarity = self.compute_representation_similarity(explicand, self.foil_encodings, self.foil_size) # Compute similarity between explicand and foil
        return corpus_similarity - foil_similarity # Compute contrastive corpus similarity by subtracting foil similarity from corpus similarity

    def compute_representation_similarity(self, explicand, encodings_dataloader, size):
        """
        Computes the representation similarity of the explicand relative to encoded data (these could be corpus or foil encodings).
        Calculation follows Equation (6), p. 8 of COCOA paper.

        Parameters:
            explicand (torch.Tensor): input data for which to compute similarity
            encodings_dataloader (torch.utils.data.DataLoader): PyTorch DataLoader containing encoded data
            size (int): number of encoded data points

        Returns:
            Representation similarity (float)
        """
        explicand_rep = self.explicand_encoder(explicand).projected_patch_embeddings # Get representation of the explicand
        explicand_rep = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(explicand_rep, (1, 1)), 1).squeeze(-1).squeeze(-1)
        representation_similarity = 0

        # For each batch of corpus encodings, compute the similarity between the explicand and the corpus
        for (x, ) in encodings_dataloader:
            x = self.compute_cosine_similarity(explicand_rep, x) # Compute cosine similarity between explicand and corpus
            representation_similarity += x.sum(dim=1)
        return representation_similarity / size  # Take mean representation similarity over all compared encodings

    # def encode_data(self, data_loader):
    #     """
    #     Encodes each corpus/foil caption in the given DataLoader.

    #     Parameters:
    #         data_loader (torch.utils.data.DataLoader): input data to encode (this is a DataLoader containing either corpus or foil captions)

    #     Returns:
    #         A concatenated sequence of encoded corpus/foil captions (torch.Tensor)
    #     """
    #     encoded_data = []
    #     for x, _ in data_loader:
    #         x_rep = self.encoder(x)
    #         encoded_data.append(x_rep)
    #     return torch.cat(encoded_data)
    
    def encode_data(self, foil_tokens):
        """
        Encodes each corpus/foil caption in the given DataLoader.

        Parameters:
            data_loader (torch.utils.data.DataLoader): input data to encode (this is a DataLoader containing either corpus or foil captions)

        Returns:
            A concatenated sequence of encoded corpus/foil captions (torch.Tensor)
        """
        encoded_data = []
        for t in foil_tokens:
            x_rep = self.encoder(t)
            encoded_data.append(x_rep)
        return torch.cat(encoded_data)
    
    def compute_cosine_similarity(self, x, y):
        """
        Computes cosine similarity between two tensors.

        Parameters:
            x, y (torch.Tensor): input arguments for cosine similarity

        Returns:
            Cosine similarity between x and y (torch.Tensor)
        """
        dot_prod = (x.unsqueeze(1) * y.unsqueeze(0)).sum(dim=-1)
        return dot_prod / (x.norm(dim=-1).unsqueeze(1) * y.norm(dim=-1).unsqueeze(0))