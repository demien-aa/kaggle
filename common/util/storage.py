import os
import pickle


def picklable(file_path, reload):
    def real_decorator(func):
        def wrapper(*args, **kwargs):
            file_name = file_path.replace('/', '-').replace('.', '-') + '.pkl'
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pickle_data', file_name)
            if os.path.exists(file) and not reload:
                result = None
                with open(file, 'rb') as input:
                    result = pickle.load(input)
                return result
            else:
                result = None
                with open(file, 'wb') as output:
                    result = func(*args, **kwargs)
                    pickle.dump(result, output)
                return result
        return wrapper
    return real_decorator


