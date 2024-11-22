FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "water_model.bin", "./"]


EXPOSE 9696

CMD ["python", "predict.py"]
