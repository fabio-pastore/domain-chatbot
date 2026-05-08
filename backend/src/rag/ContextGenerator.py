class ContextGenerator:

    @classmethod
    def generate_llm_context(cls, extracted_chunks: list[str]) -> str:
        return "\n".join(chunk for chunk in extracted_chunks)