class ContextGenerator:

    @classmethod
    def generate_llm_context(cls, extracted_chunks: set[str]) -> str:
        return "\n".join(chunk for chunk in extracted_chunks)