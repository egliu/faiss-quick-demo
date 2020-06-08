#!/bin/bash

set -euxo pipefail

docker rm -f fqd-gpu || true

echo "start local dev-gpu env"

code_root=/home/faiss-quick-demo

nvidia-docker run -d --name=fqd-gpu \
  --network=host \
  -e TZ=Asia/Shanghai \
  -v ${PWD}:${code_root} \
  -w ${code_root} \
  ${baseImage} bash -c "while true; do sleep 1000; done"

docker exec \
  -e COLUMNS="$(tput cols)" \
  -e LINES="$(tput lines)" \
  -it fqd-gpu bash