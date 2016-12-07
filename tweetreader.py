import json

class TweetReader:
    def __init__(self, filename):
        self._handle = open(filename)
        self._curr = None

    def next(self):
        self._curr = json.loads(self._handle.readline().strip())
        return self._curr

    def __iter__(self):
        orig_loc = self._handle.tell()
        self._handle.seek(0)

        for line in self._handle:
            yield json.loads(line)

        self._handle.seek(orig_loc)
