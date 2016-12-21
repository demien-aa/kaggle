import os
import pickle

from .color import blue


def picklable(file_path, reload):
    def real_decorator(func):
        def wrapper(*args, **kwargs):
            origin_func = func
            file_name = file_path.replace('/', '-').replace('.', '-') + '-' + func.__name__ + '.pkl'
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pickle_data', file_name)
            if os.path.exists(file) and not reload:
                result = None
                print(blue('Load %s from pickle' % func.__name__))
                with open(file, 'rb') as input:
                    result = pickle.load(input)
                return result
            else:
                result = None
                print(blue('Reload %s pickle obj' % func.__name__))
                with open(file, 'wb') as output:
                    result = func(*args, **kwargs)
                    pickle.dump(result, output)
                return result
        return wrapper
    return real_decorator
