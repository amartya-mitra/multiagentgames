import json
try:
   import cPickle as pickle
except:
   import pickle

class JSON:
    @classmethod
    def dump(cls, object, filename, ordered=False, desc=True):
        assert isinstance(object, dict)
        if ordered:
            object = dict(sorted(
                [(k,v) for k,v in object.items()],
                key = lambda x:x[0],
                reverse=desc
            ))
        with open(filename, 'w') as fp:
            json.dump(object, fp)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as fp:
            return json.load(fp)

class Pickle:
    @classmethod
    def dump(cls, object, filename):
        pickle.dump(object, open(filename, 'wb'))

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename, 'rb'))
