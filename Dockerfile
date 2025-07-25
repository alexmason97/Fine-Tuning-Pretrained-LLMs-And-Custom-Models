FROM python:3.12.2 

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt 

COPY . .
CMD ["pytest", "-q"]