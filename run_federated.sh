#!/usr/bin/env bash
# Launch the server and 3 hospital nodes locally.   ROUNDS=10 LOCAL_EPOCHS=2 ./run_federated.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -f certs/ca.crt ] || bash certs/generate_certs.sh
trap 'kill 0' SIGINT SIGTERM EXIT

python server.py --rounds "${ROUNDS:-10}" --local-epochs "${LOCAL_EPOCHS:-2}" &
sleep 5
for i in 0 1 2; do python client.py --node-id "$i" & done
wait %1   # finish when the server finishes its rounds
