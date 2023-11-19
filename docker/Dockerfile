FROM pytorch/pytorch

COPY ./requirements.txt .

COPY ./env.sh .

RUN pip install -r ./requirements.txt

RUN sh env.sh

CMD ['python', 'process_real_time.py']