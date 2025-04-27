FROM python:3.12.10-slim

WORKDIR /usr/src/app

RUN pip install --no-cache-dir Flask flask-wtf wtforms pandas gunicorn

# Bind working directory at runtime
# COPY flask/ /usr/src/app/

CMD ["gunicorn", "app:app", "--preload", "-b", "0.0.0.0:5000", "-w", "6"]