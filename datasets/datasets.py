import os


DEFAULT_ROOT = '/home/lzx/data/mini-ImageNet/pickle_file'



datasets = {}
def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    if kwargs.get('root_path') is None:
        kwargs['root_path'] = os.path.join(DEFAULT_ROOT)
    dataset = datasets[name](**kwargs)
    return dataset

