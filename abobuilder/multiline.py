import re
import textwrap

def clean_multiline(text: str) -> str:
    # Remove indentation
    text = textwrap.dedent(text).strip()

    # Normalize Windows/Mac line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Split into paragraphs (two or more newlines)
    paragraphs = re.split(r'\n{2,}', text)

    # Inside each paragraph: collapse single newlines → spaces
    paragraphs = [
        re.sub(r'\n+', ' ', p).strip()
        for p in paragraphs
    ]

    # Rejoin paragraphs with a blank line
    return '\n\n'.join(paragraphs)