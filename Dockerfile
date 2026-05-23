FROM python:3.11

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]