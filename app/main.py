# __pycache__ 없이..
import sys
sys.dont_write_bytecode = True

import pandas as pd

from preprocess.preprocess import fix, tokenize
from intent.intent import get_intent
from scenario.date import date
from scenario.time import times
# from scenario.dust import dust
from scenario.wise import wise
from scenario.issue import issue
from scenario.jobdam import jobdam
from scenario.basic import basic


def preprocess(speech) -> str:
    speech = fix(speech)
    speech = tokenize(speech)
    speech = fix(speech)
    return speech


def get_scenario(intent, entity) -> str:
    if intent == 'welcome':
        return basic(intent)
    elif intent == 'yes' or intent == 'no':
        return None
    elif intent == 'end':
        return basic(intent)
    elif intent == 'wisesaying':
        return wise()
    elif intent == 'issue':
        return issue()
    elif intent == 'date':
        return date()
    elif intent == 'time':
        return times()
    elif intent == 'news':
        return None
    elif intent == 'weather':
        return None # weather(entity)
    elif intent == 'dust':
        return None # dust(entity)
    elif intent == 'music':
        return None # song(entity)
    elif intent == 'wiki' or intent == 'person':
        return None # wiki(entity)
    elif intent == 'food':
        return None # restaurant(entity)
    elif intent == 'translate':
        return None # translate(entity)
    elif intent == 'jobdam':
        return jobdam()
    elif intent == 'health':
        return None
    elif intent == 'alcohol':
        return None
    elif intent == 'blood':
        return None
    elif intent == 'diabetes':
        return None
    elif intent == 'smoke':
        return None
    elif intent == 'weight':
        return None
    elif intent == 'family':
        return None
    elif intent == 'friend':
        return None
    elif intent == 'hobby':
        return None
    elif intent == 'religion':
        return None
    elif intent == 'depress':
        return None
    elif intent == 'exercise':
        return None
    else:
        return '그 기능은 아직 준비 중이에요.'



def get_speech():
    while True:
        print('User : ', sep='', end='')
        speech = preprocess(input()) # 전처리

        intent = get_intent(str(speech))
        intent = intent.strip() # 공백 제거
        print('Intent : ' + intent, sep='') # 인텐트출력
        
        entity = None
        answer = get_scenario(intent, entity)
        print('Gilbomi : ' + answer, sep='', end='\n\n')

        # entity = get_entity(intent, speech)
        # print('Entity : ' + str(entity), sep='')
        # answer = get_scenario(intent, entity)
        # print('Gilbomi : ' + answer, sep='', end='\n\n')


if __name__ == "__main__":
    sys.dont_write_bytecode=True
    print('Chatbot Test Start...')
    print('현재 인텐트 : music, health, jobdam, alcohol, blood, diabetes, smoke, weight, family, friend, hobby, religion, depress, exercise, weather, news, date, food, dust, wisesaying, translate, time, wiki, issue, person')
    # etc : welcome, no, yes, end
    get_speech()


















def get_entity(intent, speech):
    if intent == 'welcome':
        return None
    elif intent == 'yes' or intent == 'no':
        return None
    elif intent == 'end':
        return None
    elif intent == 'news':
        return None # get_news_entity(speech, False)
    elif intent == 'weather' or intent == 'dust':
        return None # get_weather_entity(speech, False)
    elif intent == 'music':
        return None # get_song_entity(speech, False)
    elif intent == 'wiki' or intent == 'person':
        return None # get_wiki_entity(speech, False)
    elif intent == 'food':
        return None # get_restaurant_entity(speech, False)
    elif intent == 'translate':
        return None # get_translate_entity(speech, False)
    elif intent == 'jobdam':
        return None # get_jobdam_entity(speech, False)
    elif intent == 'health':
        return None # get_health_entity(speech, False)
    elif intent == 'alcohol':
        return None # get_alcohol_entity(speech, False)
    elif intent == 'blood':
        return None # get_blood_entity(speech, False)
    elif intent == 'diabetes':
        return None # get_diabetes_entity(speech, False)
    elif intent == 'smoke':
        return None # get_smoke_entity(speech, False)
    elif intent == 'weight':
        return None # get_weight_entity(speech, False)
    elif intent == 'family':
        return None # get_family_entity(speech, False)
    elif intent == 'friend':
        return None # get_friend_entity(speech, False)
    elif intent == 'hobby':
        return None # get_hobby_entity(speech, False)
    elif intent == 'religion':
        return None # get_religion_entity(speech, False)
    elif intent == 'depress':
        return None # get_depress_entity(speech, False)
    elif intent == 'exercise':
        return None # get_exercise_entity(speech, False)
    else:
        return None