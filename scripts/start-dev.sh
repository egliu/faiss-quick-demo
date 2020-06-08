#!/bin/bash

set -euxo pipefail

docker rm -f fqd || true

echo "start local dev env"

code_root=/home

docker run -d --name=fqd \
  --network=host \
  -e TZ=Asia/Shanghai \
  -v ${PWD}:${code_root} \
  -w ${code_root} \
  ${baseImage} bash -c "while true; do sleep 1000; done"

docker exec \
  -e COLUMNS="$(tput cols)" \
  -e LINES="$(tput lines)" \
  -it fqd bash