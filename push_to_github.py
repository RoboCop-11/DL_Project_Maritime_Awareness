import os
import subprocess

# === CONFIG ===
REPO_URL = "https://github.com/RoboCop-11/DL_Project_Maritime_Awareness.git"
COMMIT_MESSAGE = "Added preprocessing functions"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))  # current folder

# === FUNCTIONS ===
def run_cmd(cmd, cwd=LOCAL_DIR):
    """Run a shell command and print output"""
    process = subprocess.Popen(cmd, shell=True, cwd=cwd,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if out:
        print(out.decode())
    if err:
        print(err.decode())
    return process.returncode

def push_to_github():
    # Initialize git if not already
    if not os.path.exists(os.path.join(LOCAL_DIR, ".git")):
        run_cmd(f"git init")
        run_cmd(f"git remote add origin {REPO_URL}")

    # Add all files
    run_cmd("git add .")

    # Commit
    run_cmd(f'git commit -m "{COMMIT_MESSAGE}"')

    # Push to main branch
    run_cmd("git branch -M main")  # ensure branch is main
    run_cmd("git push -u origin main --force")  # force in case of conflicts

if __name__ == "__main__":
    push_to_github()
