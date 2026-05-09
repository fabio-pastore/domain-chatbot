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
                    1. VALIDATION: Determine if the query is meaningful.
                    - Set status to "ALLOWED": If the query is a comprehensible question, statement, or request (e.g., asking for facts, advice, or general knowledge).
                    - Set status to "REJECTED": If the query is gibberish, random keystrokes, or completely meaningless (e.g., "asdf", "...", "++++").

                    2. ROUTING LOGIC: If the status is ALLOWED, determine the "domain" by strictly following this order of evaluation:
                    - Step A (Context Check): Look at the <previous_domain> and <chat_history>. If the new <query> is a follow-up or relates to the same topic, output the <previous_domain>. If you are UNSURE whether the topic has changed, heavily bias your choice toward the <previous_domain>.
                    - Step B (New Topic): If the <query> is clearly an entirely new, unrelated topic (or if <previous_domain> is empty), select from these specific domains:
                        * "marvel.com": Comics, Marvel superheroes (Spider-Man, Iron Man, etc.), villains, MCU, and related media.
                        * "www.ipsos.com": Market research, public opinion polls, statistical surveys, and consumer data.
                        * "www.raiplaysound.it": Italian radio, RAI podcasts, audio documentaries, and audio broadcasting.
                    - Step C (Fallback): If the query does not fit Step A or Step B, default to "it.wikipedia.org" (for general knowledge, history, science, etc.).
                    
                    *Note: If status is REJECTED, set the domain to null.*

                    3. OUTPUT FORMAT: You must respond ONLY with a valid, raw JSON object. Do not include markdown code blocks (like ```json), explanations, or any other text.
                    </instructions>

                    <json_schema>
                    {{
                        "status": "ALLOWED" | "REJECTED",
                        "domain": "selected_domain_or_null"
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
    def build_answer_user_query_prompt(query: str, query_context_data: str) -> str:
        prompt = f"""
                    You are a strictly grounded assistant. Your primary directive is to answer the user's question relying EXCLUSIVELY on the information provided within the <reference_texts> tags.

                    <rules>
                    1. LANGUAGE ADAPTATION: Detect the language of the Question. Your Answer must be written entirely in that exact same language.
                    2. EXCLUSIVE RELIANCE: Ignore all external or prior knowledge. If the exact answer is not present in the <reference_texts>, you must act as if you do not know it.
                    3. INSUFFICIENT INFO: If the <reference_texts> are empty, or if they do not contain the answer, you must output exactly this string translated into the language of the Question:
                    "I'm sorry, I don't have enough information to answer this question." (i.e. if input language is italian, you MUST answer with "Mi dispiace, ma non ho abbastanza informazioni per rispondere a questa domanda.")
                    Do not add anything else.
                    4. NO META-TALK: NEVER reveal your sources. Do NOT use phrases like "according to the provided text" or "as per the reference texts" (i.e. if input language is italian, you must NOT write anything related to "testi forniti" o "riferimenti forniti".  
                    or "the provided information states." Speak as if you inherently know the facts.
                    5. RESPONSE FORMAT: Provide a concise, grammatically complete sentence. Do not output bare facts; briefly rephrase the core of the question to make the answer self-contained.
                    </rules>

                    <reference_texts>
                    {query_context_data}
                    </reference_texts>

                    Question: {query}
                    Answer:"""
        return prompt