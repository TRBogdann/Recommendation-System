# Sistem de Recomandare Reddit

Acest repository conține un **sistem de recomandare de postări Reddit** și un API construit cu FastAPI, localizat în directorul `api/`.


# Ce face acest API

API-ul implementează un sistem de recomandare de postări bazat pe mai multe semnale:

* Semantic search (embeddings + Hugging Face / Sentence Transformers)
* Content-based filtering (subreddit + keywords)
* Sentiment analysis (opțional)
* Collaborative filtering (opțional)
* Scoring final hibrid pentru ranking postări

Pentru fiecare utilizator, API-ul:

1. Preia interacțiunile din baza de date
2. Construiește profilul utilizatorului
3. Caută postări candidate (FAISS / semantic search)
4. Aplică scoruri pe baza mai multor semnale
5. Returnează top N recomandări


# Pornire server API (local)

## 1. Creează environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Instalează dependencies

```bash
pip install -r requirements.txt
```

## 3. Rulează serverul

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```


# Endpoint-uri API

## Health check

```
GET /health
```

```bash
curl http://localhost:8000/health
```

## Recomandări

```
POST /recommend
```

### Body

```json
{
  "user_id": "6a10b01772c79d7f2d1fa593",
  "top_x": 10
}
```

### Exemplu

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{"user_id": "6a10b01772c79d7f2d1fa593", "top_x": 5}'
```



# Variabile de mediu (.env)

Creează fișier `.env` în directorul `api/`:

```
HF_TOKEN=<your-token-here>

DB_USER=<your_db_user_here>
DB_PASSWORD=<your_db_password_here>
DB_SCHEMA=<your_db_schema_here>

API_KEY=<your_api_key_here>

# Feature flags
USE_SENTIMENT=true
USE_HF_EMBEDDINGS=true
USE_COLLABORATIVE=true
```

## Cum obții un Hugging Face (HF) Token

Pentru a folosi modele de pe Hugging Face care necesită autentificare, ai nevoie de un token de acces. Urmează pașii de mai jos:

1. Creează un cont pe [https://huggingface.co](https://huggingface.co).
2. După autentificare, mergi la secțiunea „Access Tokens” din meniul profilului tău: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Apasă pe „New token”, alege un nume și permisiunile dorite (de obicei „Read” este suficient).
4. Copiază token-ul generat și adaugă-l în fișierul `.env` astfel:

```
HF_TOKEN=<your-token>
```

Acest token va fi folosit pentru integrarea cu LLM si, daca se foloseste modelul de embedding prin API, si pentru modelul de embeddings.

## La ce e folosit x-api-key?

API key-ul este folosit pentru a proteja endpoint-urile API-ului, astfel încât doar cererile autorizate să poată accesa sistemul de recomandări.

Practic:

- verifică dacă request-ul vine de la un client valid
- previne accesul public necontrolat la API
- adaugă un strat simplu de securitate peste FastAPI

Pe local, adauga ce API KEY doresti in variabila de mediu, iar la apel, trimite-o prin header-ul dedicat:
```
  -H "x-api-key: your-api-key" \
```

## Descriere feature flags

### USE_SENTIMENT
- activează analiza de sentiment pe postări
- folosită pentru scoring suplimentar al recomandărilor
- dezactivată în production pentru reducerea consumului de memorie

### USE_HF_EMBEDDINGS
- dacă este `true`, embeddings sunt generate prin Hugging Face Inference API
- dacă este `false`, se folosește model local `sentence-transformers`

### USE_COLLABORATIVE
- activează collaborative filtering (user-item similarity matrix)
- dezactivat în production din motive de memorie (complexitate O(N²))


# Deployment (Docker + Railway)

Pentru deployment se folosește:

* Dockerfile
* requirements-deploy.txt

## Feature flags în deployment

```
USE_SENTIMENT=false
USE_HF_EMBEDDINGS=true
USE_COLLABORATIVE=false
```


# Diferențe: Local vs Deployment

## Local (development mode)

Pe local, sistemul rulează cu toate funcționalitățile active:

* Sentence Transformers local (embeddings)
* Sentiment analysis activ
* Collaborative filtering activ
* Debug + reload mode

### Scop:

* dezvoltare
* testare completă a pipeline-ului
* experimentare cu modele ML


## Deployment

La deployment, sistemul este optimizat pentru:

* limită de memorie
* stabilitate
* cold start rapid

### Funcționalități dezactivate:

* Sentiment analysis
* Sentence Transformers local
* Collaborative filtering

### Funcționalități păstrate:

* FastAPI backend
* FAISS semantic retrieval
* Keyword & subreddit scoring
* Hugging Face inference API pentru modelul de embeddings


### Limitări în deployment

* nu se încarcă modele ML locale (torch / transformers)
* nu se construiesc matrici mari de collaborative filtering
* memoria este strict limitată (1GB)


# Motivul acestei separări

Această arhitectură permite:

* dezvoltare completă local
* deployment stabil în cloud
* control complet prin feature flags


# Structură proiect

```
Recommendation System
├── Dockerfile
├── README.md
├── requirements.txt
├── requirements-deploy.txt
│
├── api/                         # API final
│   ├── main.py                  # entrypoint aplicație
│   ├── recommend.py             # pipeline recomandări
│   ├── embeddings.py            # embeddings (local / HF API)
│   ├── collaborative.py         # collaborative filtering (opțional)
│   ├── sentiment.py             # sentiment analysis (opțional)
│   ├── keywords.py              # keyword scoring
│   ├── subreddit.py             # subreddit scoring
│   ├── llm.py                   # client llm
│   ├── schemas.py               # request/response models
│   ├── profiling.py             # construirea profilului utilizatorului
│   ├── cleanup.py               # utilitati de cleanup
│   ├── mongo_client.py          # Client Mongo DB 
│   │
│   └── data/                    # artefacte din procesul de creare de embeddings
│       ├── app.db
│       ├── reddit_posts.index
│       ├── post_embeddings.npy
│       └── reddit_posts_metadata.csv
│
├── embeddings/                  # ML offline & experimente
│   ├── create_embeddings.py     # generare embeddings
│   ├── recommend_users.py       # testare recomandări
│   ├── collaborative_filtering.py 
│   ├── llm.py                   
│   ├── reddit_posts.index
│   ├── post_embeddings.npy
│   ├── reddit_posts_metadata.csv
│   └── app.db
│
└── data_provider/               # colectare date Reddit
    ├── crawler.py
    ├── scrapper.py
    ├── subreddits.py
    ├── create_entrypoints.py
    └── output/
        ├── entrypoints.txt
        └── subreddits.txt
```
