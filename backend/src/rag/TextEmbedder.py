from fastembed import TextEmbedding
'''
from fastembed.common.model_description import PoolingType, ModelSource

TextEmbedding.add_custom_model(
    model="intfloat/multilingual-e5-small",
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf="intfloat/multilingual-e5-small"),  # can be used with an `url` to load files from a private storage
    dim=384,
    model_file="onnx/model.onnx",  # can be used to load an already supported model with another optimization or quantization, e.g. onnx/model_O4.onnx
)
'''

class TextEmbedder:

    # 1. Initialize the ONNX-optimized E5-Small
    _model = TextEmbedding(model_name="intfloat/multilingual-e5-small")

    @classmethod
    def embed_batch(
        cls, texts_to_embed: list[str], query: bool = True
    ) -> list[list[float]]:
        prefix: str = "query: " if query else "passage: "

        # IMPORTANT: E5 models require a prefix to be added to the text to be embedded
        prepared_chunks: list[str] = [f"{prefix}{chunk}" for chunk in texts_to_embed]

        # This uses all CPU cores by default and processes 256 strings at a time
        embeddings_generator = cls._model.embed(prepared_chunks)
        """
        to customize:
        
        embeddings_generator = cls._model.embed(
        prepared_chunks, 
        batch_size=64, 
        parallel=None 
        )
        """

        embeddings_list: list[list[float]] = [embedding.tolist() for embedding in embeddings_generator]
        """every single vector is forced into RAM if we use list(), this is fine in our case since we are working with small inputs (around 250'000 characters)
        problematic if we were working with large inputs (500,000+ individual chunks) or large embed models"""

        return embeddings_list
