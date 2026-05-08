from src.rag.TextEmbedder import TextEmbedder
import numpy as np

class ChunkSelector:
    
    '''
    __CHUNK_SIZE: n of characters, lower = more precision but may cut a paragraph in half, 
    higher = more context but may include irrelevant info and dilute embedding vector
    '''
    __CHUNK_SIZE: int = 900 
    __MAX_OUTPUT_LENGTH: int = 3600 #4 chunks, small models degrade significantly when the prompt gets long 
    __SIMILARITY_THRESHOLD: float = 0.75 #0.0 to 1.0

    @staticmethod
    def __calculate_cosine_similarity(v1, v2) -> float:
        """
        Calculates the cosine similarity between two numeric vectors.
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

        Args:
            page (str): The text to be chunked.

        Returns:
            list[str]: A list of text chunks.
        """
        chunks = []
        curr_chunk: str = "" 
        
        for p in page.split("\n"):
            while len(p) > cls.__CHUNK_SIZE:
                if curr_chunk:
                    chunks.append(curr_chunk)
                    curr_chunk = ""
                
                chunks.append(p[:cls.__CHUNK_SIZE])
                p = p[cls.__CHUNK_SIZE:]
                '''
                TODO: low priority, minor improvement.
                Add __CHUNK_OVERLAP = 120 
                Sliding window overlap between chunks (only those that are suddenly cut off) to avoid cutting words and sentences in half
                '''

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
        """
        Selects relevant chunks from parsed pages based on a query using cosine similarity.
    
        Args:
            query (str): The query string.
            parsed_pages (list[tuple[str, str]]): A list of tuples containing a URL and the corresponding text.

        Returns:
            dict[str, list[str]]: A dictionary mapping URLs to lists of relevant text chunks.
        """
        query_vector: list[float] = TextEmbedder.embed_text(query) 
        chunk_vecs: list[tuple[str, str, list[float]]] = []

        for page in parsed_pages:
            url = page[0]
            text_lower = page[1].lower() 
            chunks = cls.__chunking(text_lower)
            
            for c in chunks:
                c_vec: list[float] = TextEmbedder.embed_text(c)
                chunk_vecs.append((url, c, c_vec)) 

        chunk_scores: list[tuple[str, str, float]] = []
        
        for url, chunk_text, c_vec in chunk_vecs:
            score = cls.__calculate_cosine_similarity(query_vector, c_vec)
            if score >= cls.__SIMILARITY_THRESHOLD:#filter out chunks that are too dissimilar to the query, mitigating issues caused lack of useful chunks
                chunk_scores.append((url, chunk_text, score)) 
                
        chunk_scores.sort(key=lambda x: x[2], reverse=True) # score is in second position
        # scores = [elem[2] for elem in chunk_scores]
        # print(scores)
        
        selected = []
        total_chars = 0
        
        '''TODO: Re-rank after retrieval: Cosine similarity alone is noisy. 
        After fetching top-10, use a cross-encoder or simple keyword overlap score to re-rank and pick the final top-4.
        '''
        
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