import torch
from langchain.embeddings import HuggingFaceEmbeddings


def create_sbert_mpnet():
    """Wrapper function to create embeddings.

    Returns:
        HuggingFaceEmbeddings: Funciton to call to create embeddings in further methods.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on device: {device}")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device},
    )
