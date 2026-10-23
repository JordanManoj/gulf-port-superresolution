"""Read individual members of a huge remote .zip via HTTP Range requests.

Why: WorldStrat v1.1 is ~104 GB of zips on Zenodo. A zip keeps its index
(central directory) at the END of the file, so we can fetch only that index
plus the byte ranges of the samples we actually need, instead of the full archive.
"""
import io
import time
import zipfile

import requests


class HttpRangeFile(io.RawIOBase):
    """Minimal seekable, read-only file object backed by HTTP Range requests."""

    def __init__(self, url, block_size=1 << 20, session=None):
        self.url = url
        self.session = session or requests.Session()
        r = self.session.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.block_size = block_size
        self._cache = {}  # block index -> bytes (small LRU-ish cache)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _get(self, start, end):
        for attempt in range(5):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code == 206:
                    return r.content
                r.raise_for_status()
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        raise IOError(f"range request failed {start}-{end}")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        # Large reads go straight through; small reads use block cache.
        if n > self.block_size:
            data = self._get(self.pos, self.pos + n - 1)
        else:
            out = bytearray()
            p, remaining = self.pos, n
            while remaining > 0:
                b = p // self.block_size
                if b not in self._cache:
                    s = b * self.block_size
                    e = min(s + self.block_size, self.size) - 1
                    if len(self._cache) > 64:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[b] = self._get(s, e)
                blk = self._cache[b]
                off = p - b * self.block_size
                chunk = blk[off:off + remaining]
                out += chunk
                p += len(chunk)
                remaining -= len(chunk)
            data = bytes(out)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_remote_zip(url):
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20))
