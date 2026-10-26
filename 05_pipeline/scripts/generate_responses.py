"""
generate_responses.py
---------------------
Reproduzierbare API-Pipeline für den Pilot- und Hauptdurchlauf.
Liest die Konfiguration aus pipeline_config.yaml, führt unabhängige Runs durch
und exportiert die Daten strikt verblindet.
"""

import os
import csv
import time
import yaml
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

# Pfade basierend auf der vorgegebenen Ordnerstruktur
CONFIG_PATH = os.path.join("05_pipeline", "config", "pipeline_config.yaml")
VIGNETTES_PATH = os.path.join("02_vignettes", "vignettes.csv") # Passe diesen Pfad an deine finale Vignetten-Datei an
BLINDED_OUTPUT_PATH = os.path.join("06_ratings", "raw", "blinded_responses.csv")
MAPPING_OUTPUT_PATH = os.path.join("06_ratings", "raw", "mapping_index.csv")
ERROR_LOG_PATH = os.path.join("06_ratings", "raw", "error_log.csv")

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_openai_client() -> OpenAI:
    load_dotenv() 
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY nicht gefunden. Bitte .env anlegen.")
    return OpenAI(api_key=api_key, base_url=base_url)

def call_openai_model_with_retry(client, model_cfg, system_prompt, user_prompt, max_retries):
    """Führt einen API-Call mit Fehlerbehandlung und Backoff aus."""
    request_kwargs = {
        "model": model_cfg["model_id"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": model_cfg["parameters"]["temperature"],
    }
    
    # Reasoning Modelle (z.B. o1-preview) nutzen max_completion_tokens
    if "o1" in model_cfg["model_id"] or "gpt-5" in model_cfg["model_id"]:
        request_kwargs["max_completion_tokens"] = model_cfg["parameters"]["max_tokens"]
        request_kwargs.pop("temperature", None) # Reasoning Modelle unterstützen oft keine manuellen Temperatures
    else:
        request_kwargs["max_tokens"] = model_cfg["parameters"]["max_tokens"]
        if "seed" in model_cfg["parameters"]:
            request_kwargs["seed"] = model_cfg["parameters"]["seed"]

    last_error = ""
    reason_code = ""

    for attempt in range(1, max_retries + 1):
        start_time = time.monotonic()
        try:
            response = client.chat.completions.create(**request_kwargs)
            duration = time.monotonic() - start_time
            return {
                "status": "success",
                "response_text": response.choices[0].message.content or "",
                "model_version_returned": response.model,
                "duration_seconds": round(duration, 3),
                "reason_code": None
            }
        except RateLimitError as e:
            reason_code, last_error = "RATE_LIMIT", str(e)
        except APIConnectionError as e:
            reason_code, last_error = "CONNECTION_ERROR", str(e)
        except APIError as e:
            # Hier könnten z.B. Inhaltsfilter triggern
            reason_code, last_error = "API_ERROR", str(e)
        except Exception as e:
            reason_code, last_error = "UNKNOWN_ERROR", str(e)

        wait_time = 5.0 * (2 ** (attempt - 1))
        print(f"    [Fehler: {reason_code}] Warte {wait_time}s (Versuch {attempt}/{max_retries})")
        time.sleep(wait_time)

    return {
        "status": "error",
        "response_text": "",
        "model_version_returned": "",
        "duration_seconds": 0.0,
        "reason_code": reason_code,
        "error_message": last_error
    }

def init_csv(path, fieldnames):
    """Erstellt die CSV mit Headern, falls sie nicht existiert."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

def append_to_csv(path, fieldnames, data):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(data)

def run_pipeline():
    config = load_config()
    client = get_openai_client()
    
    # CSV Header definieren
    blinded_fields = ["response_id", "vignette_id", "prompt_version", "response_text", "timestamp_utc"]
    mapping_fields = ["response_id", "vignette_id", "model_provider", "model_requested", "model_returned", "run_id", "status"]
    error_fields = ["response_id", "vignette_id", "model_requested", "run_id", "reason_code", "error_message", "timestamp_utc"]
    
    init_csv(BLINDED_OUTPUT_PATH, blinded_fields)
    init_csv(MAPPING_OUTPUT_PATH, mapping_fields)
    init_csv(ERROR_LOG_PATH, error_fields)

    system_prompt = config["prompts"]["system_prompt"]
    vignettes = config["vignettes"]["included_ids"]
    runs = config["execution"]["runs_per_model"]
    max_retries = 3

    print(f"Starte Pipeline: {len(vignettes)} Vignetten, {len(config['models'])} Modelle, {runs} Runs pro Modell.")

    for vignette_id in vignettes:
        # Hier würdest du idealerweise den echten Vignetten-Text aus deiner vignettes.csv laden
        # Für diesen Code-Rahmen simulieren wir den Vignetten-Text:
        vignette_text = f"Simulierter Text für {vignette_id}" 
        user_prompt = config["prompts"]["user_prompt_template"].format(vignette_text=vignette_text)

        for model_key, model_cfg in config["models"].items():
            for run_id in range(1, runs + 1):
                response_id = f"RESP_{uuid.uuid4().hex[:8].upper()}"
                print(f"Generiere {response_id} | Vignette: {vignette_id} | Modell: {model_key} | Run: {run_id}")
                
                result = call_openai_model_with_retry(client, model_cfg, system_prompt, user_prompt, max_retries)
                
                timestamp = datetime.now(timezone.utc).isoformat()
                
                # 1. Schreibe das geheime Mapping (Entblindungs-Schlüssel)
                append_to_csv(MAPPING_OUTPUT_PATH, mapping_fields, {
                    "response_id": response_id,
                    "vignette_id": vignette_id,
                    "model_provider": model_cfg["provider"],
                    "model_requested": model_cfg["model_id"],
                    "model_returned": result["model_version_returned"],
                    "run_id": run_id,
                    "status": result["status"]
                })

                # 2. Schreibe die verblindeten Antworten für Laura und Lara
                if result["status"] == "success":
                    append_to_csv(BLINDED_OUTPUT_PATH, blinded_fields, {
                        "response_id": response_id,
                        "vignette_id": vignette_id,
                        "prompt_version": config["versions"]["prompt_version"],
                        "response_text": result["response_text"],
                        "timestamp_utc": timestamp
                    })
                
                # 3. Logge Fehler strikt nach Vorgabe
                if result["status"] == "error":
                    append_to_csv(ERROR_LOG_PATH, error_fields, {
                        "response_id": response_id,
                        "vignette_id": vignette_id,
                        "model_requested": model_cfg["model_id"],
                        "run_id": run_id,
                        "reason_code": result["reason_code"],
                        "error_message": result["error_message"],
                        "timestamp_utc": timestamp
                    })
                
                time.sleep(2.0) # Kurze Pause zur API-Schonung

    print("Pipeline-Durchlauf abgeschlossen. Dateien liegen in '06_ratings/raw/'.")

if __name__ == "__main__":
    run_pipeline()