from src.rag.TextEmbedder import TextEmbedder
import numpy as np

class ChunkSelector:
    
    '''
    __CHUNK_SIZE: n of characters, lower = more precision but may cut a paragraph in half, 
    higher = more context but may include irrelevant info and dilute embedding vector
    '''
    __CHUNK_SIZE: int = 1000 
    __MAX_OUTPUT_LENGTH: int = 12000 # 4 chunks, small models degrade significantly when the prompt gets long 
    __SIMILARITY_THRESHOLD: float = 0.0 # 0.0 to 1.0 NOTE: do we really need this? model already discriminates on whether the provided data can be used to answer the question

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
        Implements chunk overlap and respects word boundaries.

        Args:
            page (str): The text to be chunked.

        Returns:
            list[str]: A list of text chunks.
        """
        chunks = []
        overlap_chars = 150
        
        words = page.replace('\n', ' \n ').split(' ')
        words = [w for w in words if w != '']
        
        curr_chunk_words = []
        curr_len = 0
        i = 0
        
        while i < len(words):
            word = words[i]
            word_len = len(word) + (1 if curr_chunk_words else 0)
            
            if curr_len + word_len <= cls.__CHUNK_SIZE:
                curr_chunk_words.append(word)
                curr_len += word_len
                i += 1
            else:
                if curr_chunk_words:
                    chunk = " ".join(curr_chunk_words).replace(' \n ', '\n').strip()
                    if chunk:
                        chunks.append(chunk)
                    
                    overlap_len = 0
                    backtrack_i = i - 1
                    while backtrack_i >= 0 and overlap_len + len(words[backtrack_i]) + 1 <= overlap_chars:
                        overlap_len += len(words[backtrack_i]) + 1
                        backtrack_i -= 1
                    
                    if backtrack_i < i - 1:
                        i = backtrack_i + 1
                        
                    curr_chunk_words = []
                    curr_len = 0
                else:
                    # what if single word longer than CHUNK_SIZE? let's just append it anyway
                    chunks.append(word)
                    i += 1
                    curr_chunk_words = []
                    curr_len = 0
                    
        if curr_chunk_words:
            chunk = " ".join(curr_chunk_words).replace(' \n ', '\n').strip()
            if chunk:
                chunks.append(chunk)
                
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

        chunks_by_url: dict[str, list[tuple[str, float]]] = {}
        
        for url, chunk_text, c_vec in chunk_vecs:
            score = cls.__calculate_cosine_similarity(query_vector, c_vec)
            if score >= cls.__SIMILARITY_THRESHOLD: # filter out chunks that are too dissimilar to the query, mitigating issues caused lack of useful chunks
                if url not in chunks_by_url:
                    chunks_by_url[url] = []
                chunks_by_url[url].append((chunk_text, score)) 
                
        for url in chunks_by_url:
            chunks_by_url[url].sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        total_chars = 0
        
        '''TODO: Re-rank after retrieval: Cosine similarity alone is noisy. 
        After fetching top-10, use a cross-encoder or simple keyword overlap score to re-rank and pick the final top-4.
        '''
        
        # me when round is robin
        urls_available = list(chunks_by_url.keys())
        while urls_available and total_chars < cls.__MAX_OUTPUT_LENGTH:
            urls_to_remove = []
            for url in urls_available:
                if total_chars >= cls.__MAX_OUTPUT_LENGTH:
                    break
                
                chunk_text, score = chunks_by_url[url].pop(0)
                
                if total_chars + len(chunk_text) > cls.__MAX_OUTPUT_LENGTH:
                    urls_to_remove.append(url)
                    continue
                
                selected.append((url, chunk_text))
                total_chars += len(chunk_text)
                
                if not chunks_by_url[url]:
                    urls_to_remove.append(url)
            for url in set(urls_to_remove):
                if url in urls_available:
                    urls_available.remove(url)
        out: dict[str, list[str]] = {}
        for url, chunk in selected: # NOTE: selected contains pairs <url, selected_chunk>
             if url not in out:
                 out[url] = [chunk]
             else:
                 out[url].append(chunk)
        
        return out