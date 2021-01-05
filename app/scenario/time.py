import time


def times():
    H = time.strftime('%H')
    M = time.strftime('%M')
    return "현재 시각은 " + H + "시 " + M + "분입니다."