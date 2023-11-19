FROM pytorch/pytorch

COPY ./requirements.txt .

RUN pip install -r ./requirements.txt

WORKDIR /home/jovyan/app

CMD ["python", "process_real_time.py"]