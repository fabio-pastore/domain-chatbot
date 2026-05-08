from src.rag.TextEmbedder import TextEmbedder
import numpy as np
class ChunkSelector:

    __MAX_OUTPUT_LENGTH: int = 6000
    __CHUNK_SIZE: int = 1000

    @staticmethod
    def __calculate_cosine_similarity(v1, v2) -> float:
        """
        Calculates the cosine similarity between two numeric vectors.
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
        chunks = []
        curr_chunk: str = "" 
        
        for p in page.split("\n"):
            while len(p) > cls.__CHUNK_SIZE:
                if curr_chunk:
                    chunks.append(curr_chunk)
                    curr_chunk = ""
                
                chunks.append(p[:cls.__CHUNK_SIZE])
                p = p[cls.__CHUNK_SIZE:]

            separator = "\n" if curr_chunk else ""
            
            if (len(curr_chunk) + len(separator) + len(p) <= cls.__CHUNK_SIZE):
                curr_chunk += (separator + p)
            else:
                if curr_chunk: 
                    chunks.append(curr_chunk)
                curr_chunk = p 
                
        if curr_chunk:
            chunks.append(curr_chunk)
            
        return chunks
    
    @classmethod
    def select_relevant_chunks(cls, query: str, parsed_pages: list[tuple[str, str]]) -> dict[str, list[str]]:
    
        query_vector: list[float] = TextEmbedder.embed_text(query) 
        chunk_data: dict[str, list[tuple[str, list[float]]]] = {} # each url is mapped to a list of pairs <chunk, chunk_vec_representation>
        chunk_vecs: list[tuple[str, str, list[float]]] = []

        for page in parsed_pages:
            url = page[0]
            text_lower = page[1].lower() 
            chunks = cls.__chunking(text_lower)
            
            if url not in chunk_data:
                chunk_data[url] = [] 
                
            for c in chunks:
                c_vec: list[float] = TextEmbedder.embed_text(c)
                chunk_data[url].append((c, c_vec)) 
                chunk_vecs.append((url, c, c_vec)) 


        chunk_scores: list[tuple[str, str, float]] = []
        
        for url, chunk_text, c_vec in chunk_vecs:
            score = cls.__calculate_cosine_similarity(query_vector, c_vec)
            chunk_scores.append((url, chunk_text, score)) 
                
        chunk_scores.sort(key=lambda x: x[2], reverse=True) # score is in second position
        scores = [elem[2] for elem in chunk_scores]
        # print(scores)
        
        selected = []
        total_chars = 0
        for url, chunk , _ in chunk_scores:
             if total_chars + len(chunk) > cls.__MAX_OUTPUT_LENGTH:
                 break

             selected.append((url, chunk))
             total_chars += len(chunk)

        # transform into dict for output
        out: dict[str, list[str]] = {}
        for url, chunk in selected: # NOTE: selected contains pairs <url, selected_chunk>
             if url not in out:
                 out[url] = [chunk]
             else:
                 out[url].append(chunk)
        
        return out