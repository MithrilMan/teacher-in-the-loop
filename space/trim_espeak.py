"""Cut the data package of the Echogarden eSpeak NG WebAssembly build down to a few dictionaries.

    python trim_espeak.py PACKAGE_DIR OUT_DIR en,it,es,fr,de,la

PACKAGE_DIR is the unpacked npm package @echogarden/espeak-ng-emscripten (espeak-ng.js, espeak-ng.data). The
script copies espeak-ng.js with the file table of the data package rewritten, and writes an espeak-ng.data that
keeps every file except the <lang>_dict dictionaries not listed (24.0 MB -> 1.2 MB for the six above; Italian
text switches to the es, fr, de and la dictionaries for some words). Program code is unchanged.
"""
import re
import shutil
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
keep_dicts = {f"{x}_dict" for x in sys.argv[3].split(",")}
js = (src / "espeak-ng.js").read_text(encoding="utf-8")
data = (src / "espeak-ng.data").read_bytes()
m = re.search(r'loadPackage\(\{files:\[(.*?)\],remote_package_size:(\d+)\}\)', js)
files = re.findall(r'\{filename:"([^"]+)",start:(\d+),end:(\d+)\}', m.group(1))
keep, blob = [], bytearray()
for name, start, end in files:
    base = name.rsplit("/", 1)[-1]
    if base.endswith("_dict") and base not in keep_dicts:
        continue
    chunk = data[int(start):int(end)]
    keep.append(f'{{filename:"{name}",start:{len(blob)},end:{len(blob) + len(chunk)}}}')
    blob += chunk
js = js[:m.start()] + f'loadPackage({{files:[{",".join(keep)}],remote_package_size:{len(blob)}}})' + js[m.end():]
out.mkdir(parents=True, exist_ok=True)
(out / "espeak-ng.js").write_text(js, encoding="utf-8")
(out / "espeak-ng.data").write_bytes(bytes(blob))
shutil.copy2(src / "COPYING", out / "COPYING")
print(f"{len(files)} -> {len(keep)} files; {len(data)} -> {len(blob)} bytes")
