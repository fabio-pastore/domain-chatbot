class ChunkSelector:

    __MAX_OUTPUT_LENGTH: int = 6000
    __CHUNK_SIZE: int = 1000

    # chonker
    @classmethod
    def chunking(cls, page: str) -> list[str]:
        """
        Divide una pagina in chunk di dimensione __CHUNK_SIZE. Ultimo chunk può essere più corto.
        """
        chunks = []
        for i in range(0, len(page), cls.__CHUNK_SIZE):
            if (i + cls.__CHUNK_SIZE) <= len(page):
                chunks.append(page[i:i + cls.__CHUNK_SIZE])
            else:
                chunks.append(page[i:])
        
        return chunks
    
    @classmethod
    def select_relevant_chunks(cls, query: str, parsed_pages: list[tuple[str, str]]) -> dict[str, list[str]]:
        """
        Selezione semplice basata su keyword overlap.
        Niente embedding e modelli extra.
        """
        query_tokens = set(query.lower().split())
        
        scored = set() # url: [(score, chunk)]
        for page in parsed_pages:
            text_lower = page[1].lower() # extract parsed_text, [0] is url
            chunks = cls.chunking(text_lower)
            url = page[0]

            for chunk in chunks:
                score = sum(1 for token in query_tokens if token in chunk)  # conta numero di apparizioni di ogni token
                scored.add((url, score, chunk))
                
        '''
        results: list[tuple[str, str, int]] = []
        for url, value in scored.items():   
            for score, chunk in value:
                results.append((url, score, chunk))
        '''
              
        # ordina per rilevanza basandosi su score
        scored.sort(key=lambda x: x[1], reverse=True) # sort based on score
        
        # prende le pagine più rilevanti fino al limite di caratteri
        selected = []
        total_chars = 0
        for url, score, chunk in scored:
            if total_chars + cls.__CHUNK_SIZE > cls.__MAX_OUTPUT_LENGTH:
                break
            selected.append((url, chunk))
            total_chars += cls.__CHUNK_SIZE

        out: dict[str, list[str]] = {}
        for entry in selected: # NOTE: selected contains pairs <url, selected_chunk>
            if entry[0] not in out:
                out[entry[0]] = [entry[1]]
            else:
                out[entry[0]].append(entry[1])
        
        return out