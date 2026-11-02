#!/usr/bin/env bash
set -uo pipefail
born_group=$1
export BORN_GROUP="$born_group"
archive_born() {
  born_exit=$?
  export BORN_EXIT="$born_exit"
  python3 - <<'PY'
import os,json,hashlib,tarfile,base64,io
from pathlib import Path
g=int(os.environ['BORN_GROUP']);Path(f'born_group{g}_exit.json').write_text(json.dumps({'exit_status':int(os.environ['BORN_EXIT'])}))
files=list(Path('.').glob(f'born_group{g}*'))+[Path('born_production_manifest.json'),Path('production_seed_manifest.json')]
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as tf:
 for p in files:
  if p.is_file():tf.add(p,arcname=p.name)
raw=b.getvalue();enc=base64.b64encode(raw).decode();chunks=[enc[i:i+3000] for i in range(0,len(enc),3000)]
print(json.dumps({'group_id':g,'archive_sha256':hashlib.sha256(raw).hexdigest(),'archive_bytes':len(raw),'archive_chunks':len(chunks)}),flush=True)
for i,c in enumerate(chunks):print(f'BORN_ARCHIVE:{g}:{i}:{len(chunks)}:{c}',flush=True)
PY
  exit "$born_exit"
}
trap archive_born EXIT
timeout 1800 bash run_born_production.sh "$born_group"
