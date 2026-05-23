# Sistem de Recomandare Reddit

Acest repository conține un sistem de recomandare și un API FastAPI în directorul `api/`.

## Pornire server API

- Rulare locală (venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoint-uri API

- Health check:

```bash
curl http://localhost:8000/health
```

- Recomandare (POST):

Body (JSON):

```json
{
  "user_id": 12345,
  "top_x": 10
}
```

Exemplu curl:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "6a10b01772c79d7f2d1fa593", "top_x": 5}'
```

## Variabile .env

Creează un fișier `.env` în directorul `api` și adaugă variabilele necesare, de exemplu:

```
# Exemplu .env
HF_TOKEN=<your-token-here>
DB_USER=<your_db_user_here>
DB_PASSWORD=<your_db_password_here>
DB_SCHEMA=<your_db_schema_here>
```

> Nota: Pentru a utiliza llm-ul si in scriptul `embeddings/recommend_users.py` e nevoie de adaugarea unui fisier `.env` cu HF_TOKEN setat si in folderul `embeddings`, de asemenea vei avea nevoie de o conexiune MongoDB.

## Cum obții un Hugging Face (HF) Token

Pentru a folosi modele de pe Hugging Face care necesită autentificare, ai nevoie de un token de acces. Urmează pașii de mai jos:

1. Creează un cont pe [https://huggingface.co](https://huggingface.co).
2. După autentificare, mergi la secțiunea „Access Tokens” din meniul profilului tău: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Apasă pe „New token”, alege un nume și permisiunile dorite (de obicei „Read” este suficient).
4. Copiază token-ul generat și adaugă-l în fișierul `.env` astfel:

```
HF_TOKEN=<your-token>
```

Acest token va fi folosit pentru integrarea cu LLM.


