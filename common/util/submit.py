import datetime


def submit_stamp():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
