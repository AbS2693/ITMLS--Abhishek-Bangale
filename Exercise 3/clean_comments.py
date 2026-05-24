import re
from pathlib import Path

def clean_comments(code):
    """Remove excessive comments and docstrings while keeping essential ones."""
    lines = code.split('\n')
    cleaned_lines = []
    i = 0
    in_docstring = False
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Handle docstrings
        if '"""' in stripped or "'''" in stripped:
            quote = '"""' if '"""' in stripped else "'''"
            count = stripped.count(quote)
            if count == 2:
                i += 1
                continue
            elif count == 1:
                in_docstring = not in_docstring
                i += 1
                continue
        
        if in_docstring:
            i += 1
            continue
        
        # Remove inline comments (but keep important ones)
        if '#' in line and not stripped.startswith('#'):
            code_part, _, comment_part = line.rpartition('#')
            comment = comment_part.strip().lower()
            
            # List of comment patterns to remove (obvious/redundant)
            remove_patterns = [
                'load', 'get', 'create', 'train', 'evaluate', 'loss',
                'append', 'print', 'save', 'set', 'import', 'initialize',
                'define', 'add', 'compute', 'configure', 'setup',
                'forward pass', 'backward pass', 'statistics'
            ]
            
            should_remove = any(comment.startswith(p) for p in remove_patterns)
            
            if should_remove:
                cleaned_lines.append(code_part.rstrip())
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
        
        i += 1
    
    # Remove excessive blank lines
    result = []
    blank_count = 0
    for line in cleaned_lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 1:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    
    return '\n'.join(result)

# Clean all Python files in Exercise 3
base_path = Path(".")
py_files = list(base_path.glob("exercise 3.*/*/*.py"))

for py_file in py_files:
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned = clean_comments(content)
        
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        print(f"✓ Cleaned: {py_file.name}")
    except Exception as e:
        print(f"✗ Error processing {py_file.name}: {e}")

print("\nAll Python files cleaned successfully!")
