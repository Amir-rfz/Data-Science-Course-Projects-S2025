FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

WORKDIR /app/Final_project/Phase2     
RUN pip install --upgrade pip && \
    pip install --timeout 120 --retries 10 --no-cache-dir \
    -i https://pypi.org/simple -r requirements.txt
RUN pip install -r requirements.txt  

ENTRYPOINT ["python", "pipeline.py"]

