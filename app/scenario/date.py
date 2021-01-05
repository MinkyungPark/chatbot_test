from datetime import datetime

now = datetime.now()

def date():
    Y = now.strftime('%Y')
    M = now.strftime('%m')
    D = now.strftime('%d')
    return "오늘 날짜 알려드릴게요. 오늘은 " + Y + "년 " + M + "월 " + D + "일입니다."