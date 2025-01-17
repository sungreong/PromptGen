#!/bin/bash

# logs 디렉토리 생성
mkdir -p /app/logs

# 필요한 환경 변수 설정
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Streamlit 앱 실행
streamlit run prompt_flow.py 