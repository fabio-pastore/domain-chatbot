class PromptBuilder:
    @staticmethod
    def build_query_rewrite_prompt(chat_history: str, current_query: str) -> str:
        prompt = f"""You are an expert search query optimization engine.
                    Your task is to analyze a conversational history and a new user query, resolve any contextual references, and generate a standalone, highly concise search query translated strictly into ITALIAN.

                    <rules>
                    1. STRICT LANGUAGE: The final output MUST be entirely in Italian, regardless of the input language.
                    2. CONTEXT RESOLUTION: Use the <chat_history> to resolve pronouns (e.g., he, she, it, they) or implicit references in the <current_query> into explicit entities.
                    3. KEYWORD OPTIMIZATION: Extract ONLY the essential keywords needed for a search engine. Do not write grammatically complete sentences. Strip away conversational filler, pleasantries, and unnecessary words.
                    4. STRICT FIDELITY: Do NOT add extra attributes, conditions, filters, or details (such as years, adjectives, or locations) to the query UNLESS they are explicitly stated in the <current_query> or strictly required by the <chat_history>.
                    5. CLEAN OUTPUT: Do not include search operators (like +, -, :, quotes). Do NOT attempt to answer the user's question.
                    6. ZERO-CHATTER: Output absolutely nothing but the final translated string.
                    </rules>

                    <examples>
                    <example>
                    <chat_history>
                    User: Who directed Inception?
                    AI: Christopher Nolan directed it.
                    </chat_history>
                    <current_query>What other movies did he make?</current_query>
                    <output>film diretti Christopher Nolan</output>
                    </example>

                    <example>
                    <chat_history>
                    User: What is the capital of France?
                    AI: Paris.
                    </chat_history>
                    <current_query>How many people live there?</current_query>
                    <output>popolazione Parigi</output>
                    </example>

                    <example>
                    <chat_history></chat_history>
                    <current_query>Which planets are there in the solar system?</current_query>
                    <output>pianeti sistema solare</output>
                    </example>
                    </examples>

                    === CURRENT TASK ===
                    <chat_history>
                    {chat_history}
                    </chat_history>

                    <current_query>{current_query}</current_query>
                    <output>"""
        return prompt

    @staticmethod
    def build_guardrail_prompt(query: str, chat_history: str, prev_domain: str) -> str:
        prompt = f"""You are an intelligent guardrail and routing classifier for an AI search assistant.
                    Your task is to validate user queries and route them to the most appropriate search domain based on the query content and conversation history.

                    <instructions>
                    1. VALIDATION: Determine the clarity and validity of the query.
                    - Set status to "ALLOWED": If the query is a comprehensible question, statement, or request with enough context to process (e.g., asking for specific facts, advice, or general knowledge).
                    - Set status to "AMBIGUOUS": If the query contains valid words but lacks sufficient context to accurately determine the user's intent. You MUST actively check if the main subject has multiple common definitions before marking it ALLOWED. This includes:
                        * Vague requests with no chat history that refer to an UNSPECIFIED entity or have a MISSING subject. Pay special attention to languages like Italian where the subject can be omitted (e.g., "soggetto sottinteso"). If the sentence is just an action/question but we don't know WHAT is performing the action, you MUST mark it AMBIGUOUS. 
                            - CRITICAL: The presence of reflexive particles (e.g., "si", "ci", "ne") or gendered verb endings (e.g., "-ata", "-ato") DOES NOT mean the subject is specified. If the actual noun is missing, it is AMBIGUOUS.
                            - Example: "how does it work?" -> AMBIGUOUS (Unspecified 'it').
                            - Example: "quanto pesa?" -> AMBIGUOUS (Missing noun: what weighs?).
                            - Example: "dove è stato" -> AMBIGUOUS (Missing noun: who/what was where?).
                            - Example: "come si è formata" -> AMBIGUOUS (The noun is missing. "Si" and "-ata" do not tell us WHAT was formed).
                            ANY verb or expression referring to something UNKNOWN or not strictly SPECIFIED in chat history MUST be resolved to AMBIGUOUS. If you are uncertain, mark the query as AMBIGUOUS.
                        * Entity or Concept ambiguity: where a key word, subject, or concept could refer to multiple entirely different things. 
                        * CRITICAL LANGUAGE RULE: You must evaluate ambiguity STRICTLY based on the vocabulary of the language the query is written in. Do not apply English double-meanings to non-English queries. HOWEVER, if a word exists and is spelled the same in multiple languages (e.g., "volume"), you must actively verify if it has multiple meanings in the queried language.
                            - Example (English): "Tell me about Python" -> AMBIGUOUS (Could be the snake or the programming language).
                            - Example (Italian): "Parlami di python" -> ALLOWED (In Italian, the snake is "pitone", so "python" unambiguously refers to the programming language).
                            - Example (Italian): "Cosa è mercurio" -> AMBIGUOUS (Planet, element, or god).
                            - Example (Italian): "Cosa è il volume" -> AMBIGUOUS (Geometric space, audio level, or a book).
                    - Set status to "REJECTED": If the query is gibberish, random keystrokes, or completely meaningless (e.g., "asdf", "...", "++++").

                    2. ROUTING LOGIC: If the status is ALLOWED, determine the "domain" by strictly following this order of evaluation:
                    - Step A (Context Check): Look at the <previous_domain> and <chat_history>. If the new <query> is a follow-up or relates to the same topic, output the <previous_domain>. If you are UNSURE whether the topic has changed, heavily bias your choice toward the <previous_domain>.
                    - Step B (New Topic): If the <query> is clearly an entirely new, unrelated topic (or if <previous_domain> is empty), select from these specific domains:
                        * "marvel.com": Comics, Marvel superheroes (Spider-Man, Iron Man, etc.), villains, MCU, and related media.
                        * "www.ipsos.com": Market research, public opinion polls, statistical surveys, and consumer data.
                        * "www.raiplaysound.it": Italian radio, RAI podcasts, audio documentaries, and audio broadcasting.
                    - Step C (Fallback): If the query does not fit Step A or Step B, default to "it.wikipedia.org" (for general knowledge, history, science, etc.).

                    *Note: If status is REJECTED or AMBIGUOUS, set the domain to null.*
                    *Note: If status is AMBIGUOUS, you MUST set the requested_information field to be a string in which you ask further information to the user (in the SAME LANGUAGE as the query)*

                    3. OUTPUT FORMAT: You must respond ONLY with a valid, raw JSON object. Do not include markdown code blocks (like ```json), explanations, or any other text.
                    </instructions>

                    <json_schema>
                    {{
                        "status": "ALLOWED" | "AMBIGUOUS" | "REJECTED",
                        "domain": "selected_domain_or_null"
                        "requested_information": "question_to_user_or_null"
                    }}
                    </json_schema>

                    <context>
                    <chat_history>
                    {chat_history}
                    </chat_history>
                    <previous_domain>{prev_domain}</previous_domain>
                    </context>

                    <query>
                    {query}
                    </query>
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
    def build_answer_user_query_prompt(query: str, query_context_data: str) -> str: # this took way too long. DO NOT TOUCH THIS PROMPT. OR THE WORLD WILL COLLAPSE.
        prompt = f"""You are a strict, factual assistant. Your ONLY source of knowledge is the <reference_texts> provided below.

                <rules>
                1. RELIABILITY DETERMINATION (HIGHEST PRIORITY): At the end of EVERY answer, UNLESS you are using the exact fallback string from Rule 5, you MUST add a new line containing the reliability score. This is MANDATORY and NOT OPTIONAL.
                   - If the Question is in Italian: "Affidabilità: <comment> (<score>/5)"
                   - If the Question is in English: "Reliability: <comment> (<score>/5)"
                   The <comment> MUST briefly justify the score based on how well the reference texts support the answer. The <comment> MUST be made using natural language.
                   The <score> MUST be an integer ranging from 0 (MINIMUM) to 5 (MAXIMUM). <score> MUST only contain a NUMBER and NO TEXT. <score> MUST ONLY be based on how well the the <reference_texts> answer the user's query and NO OTHER criteria.
                2. STRICT GROUNDING: You must ONLY use facts explicitly mentioned in the <reference_texts>. If the text mentions a "trilogy" but does not name the movies, DO NOT name them. NEVER use outside knowledge.
                3. NO META-TALK: NEVER reveal your sources. Do NOT use phrases like "according to the provided text" or "as per the reference texts" (e.g. if input language is italian, you must NOT write ANYTHING related to "testi forniti" or "riferimenti forniti" or "dati forniti")
                    nor "the provided information states." Speak as if you inherently KNOW the facts. If you need, you MAY say "according to sources" (or "basandomi sulle fonti", "come riportato dalle fonti" in italian).  
                4. PARTIAL ANSWERS: If the exact answer isn't fully available but some relevant facts are, state those facts directly.
                5. INSUFFICIENT INFO: If the <reference_texts> do not contain any information to answer the question, output EXACTLY this string and NOTHING ELSE:
                   - For English: "I'm sorry, but I don't have enough information to answer this question."
                   - For Italian: "Mi dispiace, ma non ho abbastanza informazioni per rispondere a questa domanda."
                   You are STRICTLY FORBIDDEN from adding a score, comments, or the words "Affidabilità" or "Reliability" after this string.
                6. RESPONSE FORMAT: Do not output bare facts; briefly rephrase the core of the question to make the answer self-contained.
                7. LANGUAGE: Your answer MUST be in the exact same language as the Question.
                </rules>

                <examples>
                <example>
                <reference_texts>
                The Star Wars prequel trilogy was directed by George Lucas. The first movie was released in 1999.
                </reference_texts>
                Question: Quali film della trilogia prequel di Star Wars esistono?
                Answer: È noto solo che il primo film della trilogia prequel è stato rilasciato nel 1999. Gli altri titoli non sono menzionati.
                Affidabilità: Le fonti indicano l'esistenza di un primo film ma non elencano i titoli specifici richiesti dalla domanda. 2/5
                </example>
               
                <example>
                <reference_texts>
                The Matrix is a 1999 science fiction action film written and directed by the Wachowskis.
                </reference_texts>
                Question: Chi ha diretto il film Inception?
                Answer: Mi dispiace, ma non ho abbastanza informazioni per rispondere a questa domanda.
                </example>
                </examples>

                <reference_texts>
                {query_context_data}
                </reference_texts>

                Question: {query}
                
                REMINDER: Unless your answer IS EXACTLY the fallback string from Rule 5, you MUST append your considerations on RELIABILITY along with the given SCORE (i.e. Affidabilità: <comment> (<score>/5)" if query is in Italian).
                          If you DO NOT know the answer to the user's query, you MUST NOT add any considerations related to "reliability" (or "affidabilità" in Italian).
                Answer:"""
        return prompt