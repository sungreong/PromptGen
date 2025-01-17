FROM python:3.9-slim

# curl 설치 추가
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 필요한 파일 복사 및 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
COPY start.sh /start.sh
RUN chmod +x /start.sh

# 필요한 디렉토리 생성
RUN mkdir -p /app/logs /app/template && \
    chmod -R 777 /app && \
    chmod -R 777 /app/logs && \
    chmod -R 777 /app/template

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["/start.sh"] 