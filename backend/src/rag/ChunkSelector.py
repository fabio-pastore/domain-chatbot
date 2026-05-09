from src.rag.TextEmbedder import TextEmbedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

class ChunkSelector:
    
    '''
    __CHUNK_SIZE: n of characters, lower = more precision but may cut a paragraph in half, 
    higher = more context but may include irrelevant info and dilute embedding vector
    '''
    __CHUNK_SIZE: int = 1000 
    __MAX_OUTPUT_LENGTH: int = 8500 
    __CHUNK_OVERLAP: int = 150
    __COSINE_SIMILARITY_THRESHOLD: float = 0.1 # keep this low, reranking will do the hard work
    __CROSS_ENCODING_SIMILARITY_THRESHOLD: float = 0.3
    __TOP_K_CHUNKS: int = 20
    
    _reranker = None

    @staticmethod
    def __calculate_cosine_similarity(v1, v2) -> float:
        """
        Bi-Encoder: calculates the cosine similarity between two numeric vectors.
        Extremely fast but not very accurate.
        Args:
            v1 (list[float]): The first vector.
            v2 (list[float]): The second vector.

        Returns:
            float: The cosine similarity between the two vectors.
        """
        vec_a = np.array(v1)
        vec_b = np.array(v2)
        
        dot_product = np.dot(vec_a, vec_b)
        
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
    
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)

    # chonker
    @classmethod
    def __chunking(cls, page: str) -> list[str]:
        """
        Splits a page of text by newlines into chunks of a specified max size.
        Implements chunk overlap and respects word boundaries.

        Args:
            page (str): The text to be chunked.

        Returns:
            list[str]: A list of text chunks.
        """
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = cls.__CHUNK_SIZE,
            chunk_overlap  = cls.__CHUNK_OVERLAP,
            #separators=["\n\n", "\n", ".", " ", ""]
            #default separators are ["\n\n", "\n", " ", ""], having "." as separator can be bad since it's not only used to end a sentence (decimal numbers etc..)
        )
        return text_splitter.split_text(page)
        
    @classmethod
    def __rerank_chunks(cls, query: str, candidates: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
        """
        Re-ranks a list of candidate chunks using a Cross-Encoder.
        Extremely accurate but slower model than the Bi-Encoder.
        
        Args:
            query (str): The user's query.
            candidates: A list of tuples (url, chunk_text, cosine_score).
            
        Returns:
            A globally sorted list of re-ranked candidates.
        """
        if not candidates:
            return []
            
        if cls._reranker is None:
            from sentence_transformers import CrossEncoder # type: ignore do not remove I don't want this VScode warning thank you
            cls._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') # lazy loading
            
        # Prepare pairs for the CrossEncoder
        pairs = [[query, chunk_text] for _, chunk_text, _ in candidates]
        
        # Predict relevance scores
        scores = cls._reranker.predict(pairs)
        
        # Combine original data with new scores
        reranked_candidates = []
        for i, score in enumerate(scores):
            if score >= cls.__CROSS_ENCODING_SIMILARITY_THRESHOLD: 
                url, chunk_text, _ = candidates[i]
                reranked_candidates.append((url, chunk_text, float(score)))
            
        reranked_candidates.sort(key=lambda x: x[2], reverse=True)
        return reranked_candidates

    
    @classmethod
    def select_relevant_chunks(cls, query: str, parsed_pages: list[tuple[str, str]]) -> dict[str, list[str]]:
        """
        Selects relevant chunks globally based on cosine similarity, re-ranks them, 
        and packs them into the maximum context window.
        """
        query_vector: list[float] = TextEmbedder.embed_batch([query])
        
        all_candidates: list[tuple[str, str]] = []
        
        for page in parsed_pages:
            url = page[0]
            text_lower = page[1].lower() 
            chunks = cls.__chunking(text_lower)
            for c in chunks:
                all_candidates.append((url, c))
                
        if not all_candidates:
            return {}

        chunk_texts = [candidate[1] for candidate in all_candidates]
        chunk_vectors: list[list[float]] = TextEmbedder.embed_batch(chunk_texts)

        chunk_scores: list[tuple[str, str, float]] = []
        
        '''
        Bi-Encoder to prelinarly rank the numerous chunks, followed by Cross-Encoder to accurately rerank the top k chunks.
        '''
        
        for (url, chunk_text), c_vec in zip(all_candidates, chunk_vectors):
            score = cls.__calculate_cosine_similarity(query_vector, c_vec)
            if score >= cls.__COSINE_SIMILARITY_THRESHOLD: 
                chunk_scores.append((url, chunk_text, score)) 
                
        chunk_scores.sort(key=lambda x: x[2], reverse=True) 
        
        top_k_for_rerank = chunk_scores[:cls.__TOP_K_CHUNKS]
        reranked_scores = cls.__rerank_chunks(query, top_k_for_rerank)
        
        out: dict[str, list[str]] = {}
        total_chars = 0
        
        for url, chunk, _ in reranked_scores:
             if total_chars + len(chunk) > cls.__MAX_OUTPUT_LENGTH:
                 continue # if it doesn't fit try to keep looking for smaller chunks to include if possible

             if url not in out:
                 out[url] = []
             out[url].append(chunk)
             
             total_chars += len(chunk)
        
        return out