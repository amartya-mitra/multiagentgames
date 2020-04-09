import attr

def functiontable(cls):
  return FunctionTable(vars(cls))

@attr.s
class FunctionTable:
  dikt = attr.ib(factory=dict)
  def __getitem__(self, key):
    return self.dikt[key]
  __getattr__ = __getitem__
