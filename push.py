import os
import sys

def push(message):
    os.system("git add .")
    os.system(f'git commit -m "{message}"')
    os.system("git push origin main")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python push.py <commit_message>")
    else:
        commit_message = sys.argv[1]
        push(commit_message)