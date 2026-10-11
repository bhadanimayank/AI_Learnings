import urllib.request
import os

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
path = "shakespeare.txt"

if not os.path.exists(path):
    urllib.request.urlretrieve(url, path)

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

print("total characters:", len(text))
print(text[:300])   # peek at the first 300 characters