import glob
import subprocess
import sys
import os

cwd = os.path.abspath(os.path.dirname(__file__))
pattern = os.path.join(cwd, 'examples', '*.py')
files = sorted(glob.glob(pattern))
print(f'Found {len(files)} example scripts')
for f in files:
    print('=== Running:', os.path.relpath(f, cwd))
    r = subprocess.run([sys.executable, f])
    if r.returncode != 0:
        print('*** Exit code:', r.returncode)
print('Done')
