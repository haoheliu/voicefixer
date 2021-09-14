import os

def find_and_build(root,path):
    path = os.path.join(root, path)
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
    return path

def root_path(repo_name="voicefixer"):
    path = os.path.abspath(__file__)
    return path.split(repo_name)[0]