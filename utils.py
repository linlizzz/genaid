import pandas as pd
from dotenv import dotenv_values, load_dotenv
import os
from pathlib import Path
from typing import Dict


def read_secrets(secrets_file: str = "secrets/api_keys.env") -> dict:
    env_path = Path(secrets_file).resolve()
    if not env_path.exists():
        raise FileNotFoundError(f"{env_path} does not exist")
    
    secrets = dotenv_values(env_path)
    return secrets

def load_samples_data(path):
    db = pd.read_csv(path)
    tarinat = db["Tarina"].tolist() # Patient stories
    suositukset = db["Arvioitava hoitosuositus"].tolist() # Clinical references
    mallivastaukset = db["Mallivastaus"].tolist() # Sample answers
    return tarinat, suositukset, mallivastaukset

def load_guidelines_data(path):
    db = pd.read_csv(path)
    suositukset = db["Arvioitava hoitosuositus"].tolist()
    return suositukset

def load_paths(env_file: str = "secrets/paths.env") -> dict:
    """
    Load environment variables from a given .env file (default: paths.env).
    After calling this, you can access values via os.getenv("KEY").

    Args:
        env_file (str): path to the env file
    """
    env_path = Path(env_file).resolve()
    if not env_path.exists():
        raise FileNotFoundError(f"{env_path} does not exist")

    load_dotenv(env_path, override=True)
    return dotenv_values(env_path)

# ---------------- Prompt utils ----------------
def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def render_prompt(tpl: str, mapping: Dict[str, str]) -> str:
    out = tpl
    for k, v in mapping.items():
        out = out.replace(f"{{{{{k}}}}}", v)
    return out
