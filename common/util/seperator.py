from .color import white, blue_bg, green_bg


def single_line(input, char='#'):
    return char * 15 + ' ' * 2 + input + ' ' * 2 + char * 15


def print_header_footer(name):
    def real_decorator(func):
        def wrapper(*args, **kwargs):
            print(blue_bg('[Start] ') + white(single_line(name)))
            result = func(*args, **kwargs)
            print(green_bg('[End] ') + white(single_line(name, '-') + '\n'))
            return result
        return wrapper
    return real_decorator
