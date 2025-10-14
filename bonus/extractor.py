import re


def extract_function_name(code: str) -> str:
    code = code.strip()
    
    python_pattern = r'(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    match = re.search(python_pattern, code, re.MULTILINE)
    if match:
        return match.group(1)
    
    identifier_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b'
    matches = re.findall(identifier_pattern, code)
    if matches:
        return matches[0]
    
    words = code.split()[:3]
    return " ".join(words) if words else "unknown"
