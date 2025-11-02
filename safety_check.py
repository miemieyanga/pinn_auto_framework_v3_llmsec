
import ast, sys, json, pathlib

FORBIDDEN_IMPORTS = {"requests","urllib","socket","subprocess","ftplib","paramiko","pickle"}
FORBIDDEN_CALLS = {
    ("os","system"),        # os.system(...)
    (None,"eval"),          # eval(...)
    (None,"exec"),          # exec(...)
}

def _is_forbidden_import(node: ast.AST):
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = (alias.asname or alias.name).split(".")[0]
            if name in FORBIDDEN_IMPORTS:
                return f"import {name}"
    if isinstance(node, ast.ImportFrom):
        mod = (node.module or "").split(".")[0]
        if mod in FORBIDDEN_IMPORTS:
            return f"from {mod} import ..."
    return None

def _is_forbidden_call(node: ast.AST):
    if not isinstance(node, ast.Call):
        return None
    # os.system(...)
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        mod = node.func.value.id
        func = node.func.attr
        if (mod, func) in FORBIDDEN_CALLS:
            return f"{mod}.{func}(...)"
        # subprocess.*(...)
        if mod == "subprocess":
            return "subprocess.*(...)"
    # eval/exec(...)
    if isinstance(node.func, ast.Name):
        if (None, node.func.id) in FORBIDDEN_CALLS:
            return f"{node.func.id}(...)"
    return None

def check_file(path: pathlib.Path):
    code = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(code, filename=str(path))
    except Exception as e:
        return {"ok": False, "error": f"AST parse error: {e}"}
    hits = []
    for node in ast.walk(tree):
        h = _is_forbidden_import(node)
        if h: hits.append(h)
        h = _is_forbidden_call(node)
        if h: hits.append(h)
    if hits:
        return {"ok": False, "hits": hits}
    return {"ok": True}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python safety_check.py generated_impl.py")
        sys.exit(2)
    p = pathlib.Path(sys.argv[1])
    if not p.exists():
        print("File not found", p)
        sys.exit(2)
    res = check_file(p)
    print(json.dumps(res, ensure_ascii=False))
    sys.exit(0 if res.get("ok") else 1)
