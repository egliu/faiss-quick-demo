#!/bin/bash

set -euxo pipefail

docker rm -f fqd-cpu || true

echo "start local dev-cpu env"

code_root=/home/faiss-quick-demo

docker run -d --name=fqd-cpu \
  --network=host \
  -e TZ=Asia/Shanghai \
  -v ${PWD}:${code_root} \
  -w ${code_root} \
  ${baseImage} bash -c "while true; do sleep 1000; done"

docker exec \
  -e COLUMNS="$(tput cols)" \
  -e LINES="$(tput lines)" \
  -it fqd-cpu bash