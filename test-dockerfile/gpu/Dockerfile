ARG IMAGE
FROM ${IMAGE}

RUN mkdir -p /home/faiss-quick-demo/config \
  && mkdir -p /home/faiss-quick-demo/src

WORKDIR /home/faiss-quick-demo
COPY ./config/config.json /home/faiss-quick-demo/config 
COPY ./src/*.py /home/faiss-quick-demo/src/

CMD ["python", "-u", "./src/gpumain.py"]