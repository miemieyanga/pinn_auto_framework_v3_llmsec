\
import subprocess, os, shutil

def docker_available():
    try:
        subprocess.run(["docker","--version"], capture_output=True, text=True, timeout=5)
        return True
    except Exception:
        return False

def run_in_docker(image: str, host_dir: str, entry_py: str, timeout=120):
    cmd = [
        "docker","run","--rm",
        "--network","none",
        "--cpus","1.0",
        "--memory","1g",
        "-v", f"{os.path.abspath(host_dir)}:/work:rw",
        image,
        "python", f"/work/{entry_py}"
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr

def run_local(entry_py: str, timeout=120):
    # local (less safe); ensure no shell=True
    p = subprocess.run(["python", entry_py], capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr
