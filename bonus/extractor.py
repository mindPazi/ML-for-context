import re


def extract_function_name(code: str) -> str:
    code = code.strip()
    
    patterns = [
        r'(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        r'(?:public|private|protected|static|final|abstract|synchronized|native|\s)+[\w<>\[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        r'(?:async\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        r'(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s+)?\(',
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*function\s*\(',
        r'(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'func\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        r'fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        r'(?:public\s+|private\s+|protected\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        r'sub\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        r'(?:inline\s+|static\s+|extern\s+)?[\w<>\[\]\*]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, code, re.MULTILINE)
        if match:
            for i in range(1, match.lastindex + 1 if match.lastindex else 1):
                if match.group(i):
                    return match.group(i)
    
    identifier_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b'
    matches = re.findall(identifier_pattern, code)
    if matches:
        return matches[0]
    
    words = code.split()[:3]
    return " ".join(words) if words else "unknown"
