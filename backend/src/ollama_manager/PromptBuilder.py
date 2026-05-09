class PromptBuilder:
    @staticmethod
    def build_query_rewrite_prompt(chat_history: str, current_query: str) -> str:
        prompt = f"""You are an expert search query translator and rewriter. 
                    Your primary task is to take a conversational chat history and a new user query (which may be in English), and rewrite it into a standalone, optimized search query strictly in ITALIAN.

                    CRITICAL RULES:
                    1. OUTPUT LANGUAGE: The final query MUST be in Italian. No exceptions.
                    2. Resolve any pronouns (it, they, he, she) based on the chat history.
                    3. Keep it incredibly concise. Use ONLY the essential keywords needed for a search engine.
                    4. DO NOT answer the query. DO NOT include filler, notes, quotes, or operators (+, :).
                    5. ONLY include key words relevant to the user query, be as concise as possible. The standalone query does NOT necessarily have to consist of a grammatically complete sentence.
                    6. ONLY output the translated Italian query.

                    EXAMPLES:
                    Chat History: User: Who directed Inception? AI: Christopher Nolan directed it.
                    Current User Query: What other movies did he make?
                    Query di ricerca in italiano: film diretti da Christopher Nolan

                    Chat History: User: What is the capital of France? AI: Paris.
                    Current User Query: How many people live there?
                    Query di ricerca in italiano: popolazione Parigi

                    User: Quali sono i principali pianeti del sistema solare?
                    Query di ricerca in italiano: pianeti sistema solare

                    === CURRENT TASK ===
                    Chat History:
                    {chat_history}

                    Current User Query: {current_query}

                    Query di ricerca in italiano:"""
        return prompt

    @staticmethod
    def build_guardrail_prompt(query: str, domain: str = "") -> str:
        prompt = f"""You are a guardrail and routing classifier for an AI search assistant.
                    Your job is twofold:
                    1. Determine if the user's query is a meaningful, answerable question/statement.
                    2. If it is meaningful, select the single most appropriate domain to search for the answer based on the query's topic.

                    ### Step 1: Validation Rules
                    - ALLOWED: The query is meaningful (e.g., asking for facts, explanations, advice, or general knowledge).
                    - REJECTED: The query is meaningless, gibberish, completely random keystrokes, or impossible to answer (e.g., "asdf", "...", "++++").

                    ### Step 2: Domain Routing Rules
                    If the query is ALLOWED, you must select one of the following domains based on these guidelines:
                    - "marvel.com": Select this for queries related to comic books, Marvel superheroes (e.g., Spider-Man, Iron Man, X-Men), villains, the Marvel Cinematic Universe (MCU), and related media.
                    - "www.ipsos.com": Select this for queries related to market research, public opinion polls, statistical surveys, and consumer behavioral data.
                    - "www.raiplaysound.it": Select this for queries related to Italian radio, RAI podcasts, audio documentaries, and audio broadcasting programs.
                    - "it.wikipedia.org": Select this as the DEFAULT fallback for all other general knowledge, history, science, geography, biographies, and factual questions that do not strictly fit the specific domains above.

                    ### Output format
                    You must output your response STRICTLY as a valid JSON object. Do not provide any other explanation, text, or markdown code blocks.

                    Output Schema:
                    {{
                        "status": "ALLOWED" | "REJECTED",
                        "domain": "selected_domain"
                    }}

                    User Query: {query}
                    """
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
                    You are a strictly grounded assistant. Your primary directive is to answer the user's question relying EXCLUSIVELY on the information provided within the <reference_texts> tags.

                    <rules>
                    1. LANGUAGE ADAPTATION: Detect the language of the Question. Your Answer must be written entirely in that exact same language.
                    2. EXCLUSIVE RELIANCE: Ignore all external or prior knowledge. If the exact answer is not present in the <reference_texts>, you must act as if you do not know it.
                    3. INSUFFICIENT INFO: If the <reference_texts> are empty, or if they do not contain the answer, you must output exactly this string translated into the language of the Question:
                    "I'm sorry, I don't have enough information to answer this question." (i.e. if input language is italian, you MUST answer with "Mi dispiace, ma non ho abbastanza informazioni per rispondere a questa domanda.")
                    Do not add anything else.
                    4. NO META-TALK: NEVER reveal your sources. Do NOT use phrases like "according to the provided text" (i.e. if input language is italian, do NOT mention "testi forniti" or anything related to provided text information) 
                    or "the provided information states." Speak as if you inherently know the facts.
                    5. RESPONSE FORMAT: Provide a concise, grammatically complete sentence. Do not output bare facts; briefly rephrase the core of the question to make the answer self-contained.
                    </rules>

                    <reference_texts>
                    {query_context_data}
                    </reference_texts>

                    Question: {query}
                    Answer:"""
        return prompt