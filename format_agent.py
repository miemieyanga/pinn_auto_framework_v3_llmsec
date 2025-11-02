
import re, ast, textwrap

def _strip_template_markers(code: str) -> str:
    # Remove leftover <<<TAG>>> markers lines safely
    code = re.sub(r'^\s*<<<.*?>>>.*$', '', code, flags=re.MULTILINE)
    return code

def _normalize_newlines(code: str) -> str:
    return code.replace('\r\n','\n').replace('\r','\n')

def _fix_common_hparams_block(code: str) -> str:
    # If someone mistakenly pasted assignments into HYPERPARAMS dict, try to repair.
    # Find "HYPERPARAMS = { ... }" and ensure it's a JSON-like dict with "key": value lines.
    m = re.search(r'HYPERPARAMS\s*=\s*\{(.*?)\}', code, flags=re.DOTALL)
    if not m:
        return code
    inner = m.group(1)
    # If inner contains lines like "epochs = 5000", convert to '"epochs": 5000,'
    def repl_assign_to_dict(match):
        k = match.group(1).strip()
        v = match.group(2).strip()
        return f'"{k}": {v},'
    inner_fixed = re.sub(r'^\s*([A-Za-z_]\w*)\s*=\s*([^\n#]+)$', repl_assign_to_dict, inner, flags=re.MULTILINE)
    # Ensure trailing commas and strip dup commas
    inner_fixed = re.sub(r',\s*,', ',', inner_fixed)
    code = code[:m.start(1)] + inner_fixed + code[m.end(1):]
    return code

def _deterministic_reindent(code: str) -> str:
    # Use textwrap.dedent to remove accidental over-indentation
    code = textwrap.dedent(code)
    # Replace tabs with 4 spaces
    code = code.expandtabs(4)
    # Normalize spaces around def/class
    code = re.sub(r'^\s+(class|def)\s', r'\1 ', code, flags=re.MULTILINE)
    return code

def _maybe_apply_black(code: str) -> str:
    try:
        import black
        return black.format_str(code, mode=black.Mode())
    except Exception:
        return code

def format_and_validate(code: str, use_black: bool = True) -> dict:
    original = code
    code = _normalize_newlines(code)
    code = _strip_template_markers(code)
    code = _fix_common_hparams_block(code)
    code = _deterministic_reindent(code)
    if use_black:
        code = _maybe_apply_black(code)

    # Final AST check
    try:
        ast.parse(code, filename="generated_impl.py")
        ok = True
        err = ""
    except Exception as e:
        ok = False
        err = str(e)
    return {"ok": ok, "error": err, "code": code, "changed": code != original}
