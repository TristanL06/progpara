FROM python:3.8.5-slim-buster

WORKDIR /app
#RUN apk add --no-cache python3 py3-pip

COPY ./ .
RUN apt-get update && apt-get install -y openssh-client
RUN pip install -r requirements.txt 
# RUN pip install --no-cache-dir -r requirements.txt

COPY .ssh/ /root/.ssh/

# Assurez-vous que les permissions de la clé SSH sont correctes
RUN chmod 600 /root/.ssh/boole


CMD ["python3", "test.py"]
