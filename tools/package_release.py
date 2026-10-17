import base64
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
PACK=ROOT/'veyra2-10m'
generated=PACK/'scripts/generated'
source=json.loads((PACK/'scripts/MODEL_SOURCE.json').read_text(encoding='utf-8'))
training=json.loads((ROOT/'training/training_report.json').read_text(encoding='utf-8'))
manifest=json.loads((PACK/'manifest.json').read_text(encoding='utf-8'))
model_check=json.loads((PACK/'MODEL_VALIDATION.json').read_text(encoding='utf-8'))
command_check=json.loads((PACK/'COMMAND_VALIDATION.json').read_text(encoding='utf-8'))
assert model_check['passed']==model_check['total']
assert model_check['defaultSamplingPassed']==model_check['defaultSamplingTotal']
assert all(command_check.values())
assert manifest['metadata']['authors']==['ILwHee']
assert manifest['metadata']['url']=='https://www.youtube.com/@Datkiri001'
assert manifest['header']['version']==[1,1,0]
assert manifest['capabilities']==['script_eval']
assert manifest['dependencies']==[{'module_name':'@minecraft/server','version':'beta'}]
assert source['fineTune']['modelSha256']==training['modelSha256']
assert hashlib.sha256((ROOT/'training/adapter.safetensors').read_bytes()).hexdigest()==training['adapterSha256']
def decode(pattern):
    return b''.join(base64.b64decode(re.search(r'export default "([A-Za-z0-9+/=]+)"',p.read_text(encoding='utf-8'))[1])
                    for p in sorted(generated.glob(pattern)))
weights=decode('weights_[0-9]*.js')
assert hashlib.sha256(weights).hexdigest()==source['weightsSha256']
meta=json.loads(re.search(r'export const PREFIX_META = (.*);',(generated/'prefix_data.js').read_text(encoding='utf-8'))[1])
prefix=decode('prefix_[0-9]*.js')
assert hashlib.sha256(prefix).hexdigest()==meta['sha256']
assert meta['modelId']==source['weightsSha256']
for name in ['LICENSE','NOTICE']:
    assert (ROOT/name).is_file()
for name in ['LICENSE-APACHE-2.0.txt','NOTICE']:
    assert (PACK/name).is_file()
for path in ROOT.rglob('*'):
    if not path.is_file() or path.suffix.lower() in {'.mcpack','.mcaddon','.zip'} or any(part in {'.git','dist','checkpoints','__pycache__','node_modules'} for part in path.relative_to(ROOT).parts):
        continue
    data=path.read_bytes()
    assert (b'github'+b'_pat_') not in data,'Credential pattern found in source'
    assert not re.search(rb'gh[pousr]_[A-Za-z0-9_]{24,}',data),'Credential pattern found in source'
(ROOT/'dist').mkdir(exist_ok=True)
target=ROOT/'dist/veyra2-ilwhee-1.1.0.mcpack'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(PACK.rglob('*')):
        if path.is_file() and path.suffix.lower() not in {'.mcpack','.mcaddon','.zip'}:
            archive.write(path,path.relative_to(PACK).as_posix())
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert 'manifest.json' in archive.namelist()
digest=hashlib.sha256(target.read_bytes()).hexdigest()
(ROOT/'dist/SHA256SUMS.txt').write_text(f'{digest}  {target.name}\n',encoding='ascii')
print(json.dumps({'file':str(target),'bytes':target.stat().st_size,'sha256':digest}))
