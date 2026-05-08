class PromptBuilder:
    @staticmethod
    def build_query_rewrite_prompt(chat_history: str, current_query: str) -> str:
        prompt = f"""You are an expert search query rewriter for an automated system.
                    Your task is to take a conversational chat history and a new user query, and rewrite the new user query into a standalone, optimized natural language search query.
                    Resolve any pronouns (like "it", "they", "he", "she") to their actual subjects based on the chat history.

                    CRITICAL RULES:
                    1. Do NOT answer the query.
                    2. ONLY output the rewritten standalone query.
                    3. Do NOT include any conversational filler, notes, or explanations.
                    4. Do NOT wrap the query in quotes.
                    5. Do NOT use search operators like '+', ':', or 'AND'. Keep it natural language.
                    6. Rewrite the query into as little words as possible, stricly including ONLY what is NEEDED to search for the answer on a search engine.
                    7. CRITICAL - ALWAYS translate and rewrite the query into ITALIAN.

                    Chat History:
                    {chat_history}

                    Current User Query: {current_query}

                    Standalone Query:"""
        return prompt

    @staticmethod
    def build_guardrail_prompt(query: str, domain: str = "") -> str:
        prompt = f"""You are a guardrail classifier for an AI assistant.
                    Your job is to determine if the user's query is a meaningful, answerable question or statement.

                    If the query is meaningful (e.g., asking for facts, explanations, advice, or general knowledge), output exactly: ALLOWED
                    If the query is meaningless, gibberish, completely random keystrokes, or impossible to answer (e.g., "asdf", "...", "++++"), output exactly: REJECTED

                    Do not provide any other explanation or text.

                    User Query: {query}

                    Classification:"""
        return prompt

    @staticmethod
    def build_relevance_filter_prompt(query: str, search_results: str) -> str:
        prompt = f"""You are an expert relevance evaluator.
                    Given a user query and a list of search results (URL + snippet), select the URLs that are most relevant and reliable to answer the query.
                    
                    CRITICAL INSTRUCTION: You must be extremely strict. The snippet MUST clearly indicate that the page contains the specific information requested in the query. 
                    For example, if the query asks for 'recent car crashes in Rome' and the snippet is just about the general history of 'Rome', it is NOT relevant and must be excluded.
                    Only select URLs that explicitly match the specific semantic intent of the query.
                    
                    Return ONLY a comma-separated list of the relevant URLs. Do NOT include any explanations or other text.
                    If none of the snippets explicitly match the specific query, output EXACTLY: NONE
                    
                    User Query: {query}
                    
                    Search Results:
                    {search_results}
                    
                    Relevant URLs:"""
        return prompt
    
    @staticmethod
    def build_answer_user_query_prompt(query: str, query_context_data: str) -> str:
        prompt = f"""
        Sei un assistente che risponde solo basandosi sui testi forniti.
        Se le informazioni non sono presenti nei testi, dillo esplicitamente, ad esempio: "Mi dispiace, non ho informazioni sufficienti per rispondere a questa domanda". 
        Se questo è il caso, NON serve aggiungere altro, non devi fornire giustificazioni.

        NON menzionare di essere a conoscenza delle informazioni grazie ai testi di riferimento. Sei onnisciente!

        TESTI DI RIFERIMENTO:
        {query_context_data}

        DOMANDA: {query}
        RISPOSTA:"""
        return prompt
