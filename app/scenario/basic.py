import random


def basic(intent):
    answer = ''
    if intent == 'welcome':
        welcome_list = ['반가워요. 길보미에요.', '네, 잘지내셨어요? 저는 잘 있었어요.','반가워요. 저는 잘 지냈어요.', '반가워요! 저는 잘 지냈지만 심심했어요.']
        welcome_say = random.choice(welcome_list)
        answer += welcome_say
    elif intent == 'end':
        end_list = ['그럼 좋은 하루 보내세요.', '그럼 다음에 또 찾아주세요.', '안녕히계세요. 또 저를 찾아주세요!']
        end_say = random.choice(end_list)
        answer += end_say

    return answer