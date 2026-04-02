from google.colab import drive, userdata
import os
import sys

def initialize_project():
    """
    Mounts drive, authenticates Git, and sets up system paths.
    """
    # 1. Mount Drive
    if not os.path.exists('/content/drive'):
      drive.mount('/content/drive')

    # 2. Retrieve Private Secrets
    try:
        GIT_TOKEN = userdata.get('REPO_TOKEN')
        GIT_NAME = userdata.get('GIT_NAME')
        REPO_NAME = "CNN-vs-VIT-Agribench"
        GIT_EMAIL = userdata.get('GIT_EMAIL')
    except Exception:
        print("Error: Add all required KEYS to the Colab Secrets tab first!")
        return

    # 3. Project Paths
    # Adjust this path if your shared folder name is different
    PROJECT_ROOT = f"/content/drive/MyDrive/Deep-Learning-Final-Project/{REPO_NAME}"
    
    if PROJECT_ROOT not in sys.path:
        sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
    
    os.chdir(PROJECT_ROOT)

    # 4. Safe Git Configuration
    os.system(f"git remote set-url origin https://{GIT_TOKEN}@github.com/{GIT_NAME}/{REPO_NAME}.git")
    os.system(f"git config --global user.email '{GIT_EMAIL}'")
    os.system(f"git config --global user.name '{GIT_NAME}'")

    print(f"Authenticated as: {GIT_NAME}")
    print(f"Working Directory: {os.getcwd()}")