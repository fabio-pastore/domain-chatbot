class ContextGenerator:

    @classmethod
    def generate_llm_context(cls, extracted_chunks: list[str]) -> str:
        """
        Generates a context string from a list of text chunks.

        Args:
            extracted_chunks: A list of text chunks to be joined into a single string.

        Returns:
            A single string containing all the chunks joined by newline characters.
        """
        return "\n".join(chunk for chunk in extracted_chunks)