import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import argparse
import sys
import pickle
import time
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

# Data_path='dataset/Test_now.txt'
Data_path='dataset/Test3_500_0814.txt'
w=open("Result3_500.txt",'a')
w2=open("Result3_500_sentence.txt","a")
w3 = open("Result3_vector.txt", "a")
time_end = time.time()
Default_where = ['HERE', 'nearby']

def filling_default (query):
    result_parse = query.replace(' ', ':').split(':')
    print(result_parse)
    matching = [ num for num,s in enumerate(result_parse) if 'PH' in s]
    if matching is not '':
        for i in range(len(matching)):
            if result_parse[matching[i]-1] == 'WHERE':
                query = query.replace(result_parse[matching[i]], 'HERE')
            if result_parse[matching[i]-1] == 'WHEN':
                query = query.replace(result_parse[matching[i]], 'NOW')
        if query[-1] !=']':
            query = query +']'
    return query

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())

if len(sys.argv) > 1:
    lines = sys.argv[1]
else:
    lines =  "What's the weather forecast for this afternoon?"


def process_to_IDs_in_sparse_format(sp, sentences):
    # An utility method that processes sentences with the sentence piece processor
    # 'sp' and returns the results in tf.SparseTensor-similar format:
    # (values, indices, dense_shape)
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values=[item for sublist in ids for item in sublist]
    indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)

def load_data(file_name):
    data =  [line.rstrip() for line in list(open(file_name, "r").readlines())]
    x_text = []
    y_script = []
    y_category = []

    for i, sentence in enumerate(data):
        split_result = sentence.split("||")
        x_text.append(split_result[1])
        y_script.append(split_result[3])
        y_category.append(split_result[2])

    return [x_text, y_script, y_category]


one = [
    "What's the WASHINGTON for Time?",
    "What's the SCHOOL for Time?",
    "What's the MEETING for Time?",
    "What's the MONALISA for Time?",
    "What's the NOTEBOOK for Time?",
    "What's the SOMETHING for Time?",

    "Navigate to WASHINGTON.",
    "Navigate to SOMETHING.",
    "Navigate to SCHOOL.",
    "Navigate to MONALISA.",
    "Navigate to DAVID.",

    "What's my WASHINGTON to destination?",
    "What's my SOMETHING to destination?",
    "What's my SCHOOL to destination?",

    "Show me alternative WASHINGTON.",
    "Show me alternative SOMETHING.",
    "Show me alternative SCHOOL.",
    "Show me alternative MONALISA.",

    "Reroute using WASHINGTON.",
    "Reroute using SOMETHING.",
    "Reroute using SCHOOL.",
    "Reroute using MONALISA.",

    "Drive to WASHINGTON.",
    "Drive to SOMETHING.",
    "Drive to SCHOOL.",
    "Drive to MONALISA.",

    "What's my WASHINGTON?",
    "What's my SOMETHING?",
    "What's my SCHOOL?",
    "What's my MONALISA?",

    "Can I make Time's 10am WASHINGTON without recharging?",
    "Can I make Time's 10am SOMETHING without recharging?",
    "Can I make Time's 10am SCHOOL without recharging?",
    "Can I make Time's 10am MEETING without recharging?",

    "Will it rain Time in WASHINGTON?",
    "Will it rain Time in SOMETHING?",
    "Will it rain Time in SCHOOL?",
    "Will it rain Time in MEETING?",

    "How long can I go?",
    "How far I can go?",
    "How much longer can I go?"
]

one_class_id = [
    0, 0, 0, 0, 0, 0,
    8, 8, 8, 8, 8,
    9, 9, 9,
    10, 10, 10, 10,
    11, 11, 11, 11,
    13, 13, 13, 13,
    14, 14, 14, 14,
    15, 15, 15, 15,
    18, 18, 18, 18,
    14, 14, 14
]
two = [
    "What's the WASHINGTON like on my WASHINGTON?",
    "What's the WASHINGTON like on my SOMETHING?",
    "What's the SOMETHING like on my WASHINGTON?",
    "What's the SOMETHING like on my SOMETHING?",

    "Can you find me a WASHINGTON with WASHINGTON nearby?",
    "Can you find me a WASHINGTON with SOMETHING nearby?",
    "Can you find me a WASHINGTON with SCHOOL nearby?",
    "Can you find me a SOMETHING with WASHINGTON nearby?",
    "Can you find me a SOMETHING with SOMETHING nearby?",
    "Can you find me a SOMETHING with SCHOOL nearby?",
    "Can you find me a SCHOOL with WASHINGTON nearby?",
    "Can you find me a SCHOOL with SOMETHING nearby?",
    "Can you find me a SCHOOL with SCHOOL nearby?",

    "Find a WASHINGTON along WASHINGTON.",
    "Find a WASHINGTON along SOMETHING.",
    "Find a WASHINGTON along SCHOOL.",
    "Find a SOMETHING along WASHINGTON.",
    "Find a SOMETHING along SOMETHING.",
    "Find a SOMETHING along SCHOOL.",
    "Find a SCHOOL along WASHINGTON.",
    "Find a SCHOOL along SOMETHING.",
    "Find a SCHOOL along SCHOOL.",

    "Find the cheapest WASHINGTON within Distance of my WASHINGTON.",
    "Find the cheapest WASHINGTON within Distance of my SOMETHING.",
    "Find the cheapest WASHINGTON within Distance of my SCHOOL.",
    "Find the cheapest SOMETHING within Distance of my WASHINGTON.",
    "Find the cheapest SOMETHING within Distance of my SOMETHING.",
    "Find the cheapest SOMETHING within Distance of my SCHOOL.",
    "Find the cheapest SCHOOL within Distance of my WASHINGTON.",
    "Find the cheapest SCHOOL within Distance of my SOMETHING.",
    "Find the cheapest SCHOOL within Distance of my SCHOOL.",

    "Are there any WASHINGTON on my WASHINGTON?",
    "Are there any WASHINGTON on my SOMETHING?",
    "Are there any WASHINGTON on my SCHOOL?",
    "Are there any WASHINGTON on my MONALISA?",
    "Are there any WASHINGTON on my NOTEBOOK?",

    "Are there any SOMETHING on my WASHINGTON?",
    "Are there any SOMETHING on my SOMETHING?",
    "Are there any SOMETHING on my SCHOOL?",
    "Are there any SOMETHING on my MONALISA?",
    "Are there any SOMETHING on my NOTEBOOK?",

    "Are there any SCHOOL on my WASHINGTON?",
    "Are there any SCHOOL on my SOMETHING?",
    "Are there any SCHOOL on my SCHOOL?",
    "Are there any SCHOOL on my MONALISA?",
    "Are there any SCHOOL on my NOTEBOOK?",

    "Are there any MONALISA on my WASHINGTON?",
    "Are there any MONALISA on my SOMETHING?",
    "Are there any MONALISA on my SCHOOL?",
    "Are there any MONALISA on my MONALISA?",
    "Are there any MONALISA on my NOTEBOOK?",

    "Are there any NOTEBOOK on my WASHINGTON?",
    "Are there any NOTEBOOK on my SOMETHING?",
    "Are there any NOTEBOOK on my SCHOOL?",
    "Are there any NOTEBOOK on my MONALISA?",
    "Are there any NOTEBOOK on my NOTEBOOK?"

]

two_class_id = [
    1, 1, 1, 1,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5,
    17, 17, 17, 17, 17,
    17, 17, 17, 17, 17,
    17, 17, 17, 17, 17,
    17, 17, 17, 17, 17
]
three = [
    "Show me a WASHINGTON on WASHINGTON and WASHINGTON.",
    "Show me a WASHINGTON on WASHINGTON and SOMETHING.",
    "Show me a WASHINGTON on WASHINGTON and SCHOOL.",
    "Show me a WASHINGTON on SOMETHING and WASHINGTON.",
    "Show me a WASHINGTON on SOMETHING and SCHOOL.",
    "Show me a WASHINGTON on SOMETHING and SOMETHING.",
    "Show me a WASHINGTON on SCHOOL and WASHINGTON.",
    "Show me a WASHINGTON on SCHOOL and SOMETHING.",
    "Show me a WASHINGTON on SCHOOL and SCHOOL.",

    "Show me a SOMETHING on WASHINGTON and WASHINGTON.",
    "Show me a SOMETHING on WASHINGTON and SOMETHING.",
    "Show me a SOMETHING on WASHINGTON and SCHOOL.",
    "Show me a SOMETHING on SOMETHING and WASHINGTON.",
    "Show me a SOMETHING on SOMETHING and SCHOOL.",
    "Show me a SOMETHING on SOMETHING and SOMETHING.",
    "Show me a SOMETHING on SCHOOL and WASHINGTON.",
    "Show me a SOMETHING on SCHOOL and SOMETHING.",
    "Show me a SOMETHING on SCHOOL and SCHOOL.",

    "Show me a SCHOOL on WASHINGTON and WASHINGTON.",
    "Show me a SCHOOL on WASHINGTON and SOMETHING.",
    "Show me a SCHOOL on WASHINGTON and SCHOOL.",
    "Show me a SCHOOL on SOMETHING and WASHINGTON.",
    "Show me a SCHOOL on SOMETHING and SCHOOL.",
    "Show me a SCHOOL on SOMETHING and SOMETHING.",
    "Show me a SCHOOL on SCHOOL and WASHINGTON.",
    "Show me a SCHOOL on SCHOOL and SOMETHING.",
    "Show me a SCHOOL on SCHOOL and SCHOOL.",

    "Okay, can you find me a WASHINGTON on my WASHINGTON that has a WASHINGTON?",
    "Okay, can you find me a WASHINGTON on my WASHINGTON that has a SOMETHING?",
    "Okay, can you find me a WASHINGTON on my WASHINGTON that has a SCHOOL?",
    "Okay, can you find me a WASHINGTON on my SOMETHING that has a WASHINGTON?",
    "Okay, can you find me a WASHINGTON on my SOMETHING that has a SOMETHING?",
    "Okay, can you find me a WASHINGTON on my SOMETHING that has a SCHOOL?",
    "Okay, can you find me a WASHINGTON on my SCHOOL that has a WASHINGTON?",
    "Okay, can you find me a WASHINGTON on my SCHOOL that has a SOMETHING?",
    "Okay, can you find me a WASHINGTON on my SCHOOL that has a SCHOOL?",

    "Okay, can you find me a SOMETHING on my WASHINGTON that has a WASHINGTON?",
    "Okay, can you find me a SOMETHING on my WASHINGTON that has a SOMETHING?",
    "Okay, can you find me a SOMETHING on my WASHINGTON that has a SCHOOL?",
    "Okay, can you find me a SOMETHING on my SOMETHING that has a WASHINGTON?",
    "Okay, can you find me a SOMETHING on my SOMETHING that has a SOMETHING?",
    "Okay, can you find me a SOMETHING on my SOMETHING that has a SCHOOL?",
    "Okay, can you find me a SOMETHING on my SCHOOL that has a WASHINGTON?",
    "Okay, can you find me a SOMETHING on my SCHOOL that has a SOMETHING?",
    "Okay, can you find me a SOMETHING on my SCHOOL that has a SCHOOL?",

    "Okay, can you find me a SCHOOL on my WASHINGTON that has a WASHINGTON?",
    "Okay, can you find me a SCHOOL on my WASHINGTON that has a SOMETHING?",
    "Okay, can you find me a SCHOOL on my WASHINGTON that has a SCHOOL?",
    "Okay, can you find me a SCHOOL on my SOMETHING that has a WASHINGTON?",
    "Okay, can you find me a SCHOOL on my SOMETHING that has a SOMETHING?",
    "Okay, can you find me a SCHOOL on my SOMETHING that has a SCHOOL?",
    "Okay, can you find me a SCHOOL on my SCHOOL that has a WASHINGTON?",
    "Okay, can you find me a SCHOOL on my SCHOOL that has a SOMETHING?",
    "Okay, can you find me a SCHOOL on my SCHOOL that has a SCHOOL?",

    "Find WASHINGTON near destination that accepts WASHINGTON and has a WASHINGTON.",
    "Find WASHINGTON near destination that accepts WASHINGTON and has a SOMETHING.",
    "Find WASHINGTON near destination that accepts WASHINGTON and has a SCHOOL.",
    "Find WASHINGTON near destination that accepts SOMETHING and has a WASHINGTON.",
    "Find WASHINGTON near destination that accepts SOMETHING and has a SOMETHING.",
    "Find WASHINGTON near destination that accepts SOMETHING and has a SCHOOL.",
    "Find WASHINGTON near destination that accepts SCHOOL and has a WASHINGTON.",
    "Find WASHINGTON near destination that accepts SCHOOL and has a SOMETHING.",
    "Find WASHINGTON near destination that accepts SCHOOL and has a SCHOOL.",

    "Find SOMETHING near destination that accepts WASHINGTON and has a WASHINGTON.",
    "Find SOMETHING near destination that accepts WASHINGTON and has a SOMETHING.",
    "Find SOMETHING near destination that accepts WASHINGTON and has a SCHOOL.",
    "Find SOMETHING near destination that accepts SOMETHING and has a WASHINGTON.",
    "Find SOMETHING near destination that accepts SOMETHING and has a SOMETHING.",
    "Find SOMETHING near destination that accepts SOMETHING and has a SCHOOL.",
    "Find SOMETHING near destination that accepts SCHOOL and has a WASHINGTON.",
    "Find SOMETHING near destination that accepts SCHOOL and has a SOMETHING.",
    "Find SOMETHING near destination that accepts SCHOOL and has a SCHOOL.",

    "Find SCHOOL near destination that accepts WASHINGTON and has a WASHINGTON.",
    "Find SCHOOL near destination that accepts WASHINGTON and has a SOMETHING.",
    "Find SCHOOL near destination that accepts WASHINGTON and has a SCHOOL.",
    "Find SCHOOL near destination that accepts SOMETHING and has a WASHINGTON.",
    "Find SCHOOL near destination that accepts SOMETHING and has a SOMETHING.",
    "Find SCHOOL near destination that accepts SOMETHING and has a SCHOOL.",
    "Find SCHOOL near destination that accepts SCHOOL and has a WASHINGTON.",
    "Find SCHOOL near destination that accepts SCHOOL and has a SOMETHING.",
    "Find SCHOOL near destination that accepts SCHOOL and has a SCHOOL.",

    "Find MONALISA near destination that accepts WASHINGTON and has a WASHINGTON.",
    "Find MONALISA near destination that accepts WASHINGTON and has a SOMETHING.",
    "Find MONALISA near destination that accepts WASHINGTON and has a SCHOOL.",
    "Find MONALISA near destination that accepts SOMETHING and has a WASHINGTON.",
    "Find MONALISA near destination that accepts SOMETHING and has a SOMETHING.",
    "Find MONALISA near destination that accepts SOMETHING and has a SCHOOL.",
    "Find MONALISA near destination that accepts SCHOOL and has a WASHINGTON.",
    "Find MONALISA near destination that accepts SCHOOL and has a SOMETHING.",
    "Find MONALISA near destination that accepts SCHOOL and has a SCHOOL.",

    "Find NOTEBOOK near destination that accepts WASHINGTON and has a WASHINGTON.",
    "Find NOTEBOOK near destination that accepts WASHINGTON and has a SOMETHING.",
    "Find NOTEBOOK near destination that accepts WASHINGTON and has a SCHOOL.",
    "Find NOTEBOOK near destination that accepts SOMETHING and has a WASHINGTON.",
    "Find NOTEBOOK near destination that accepts SOMETHING and has a SOMETHING.",
    "Find NOTEBOOK near destination that accepts SOMETHING and has a SCHOOL.",
    "Find NOTEBOOK near destination that accepts SCHOOL and has a WASHINGTON.",
    "Find NOTEBOOK near destination that accepts SCHOOL and has a SOMETHING.",
    "Find NOTEBOOK near destination that accepts SCHOOL and has a SCHOOL.",

    "What's the SOMETHING"
]

three_class_id = [
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7

]

four = [
    "Find WASHINGTON near WASHINGTON that accepts WASHINGTON and has a WASHINGTON.",
    "Find WASHINGTON near WASHINGTON that accepts WASHINGTON and has a SOMETHING.",
    "Find WASHINGTON near WASHINGTON that accepts WASHINGTON and has a SCHOOL.",
    "Find WASHINGTON near WASHINGTON that accepts SOMETHING and has a WASHINGTON.",
    "Find WASHINGTON near WASHINGTON that accepts SOMETHING and has a SOMETHING.",
    "Find WASHINGTON near WASHINGTON that accepts SOMETHING and has a SCHOOL.",
    "Find WASHINGTON near WASHINGTON that accepts SCHOOL and has a WASHINGTON.",
    "Find WASHINGTON near WASHINGTON that accepts SCHOOL and has a SOMETHING.",
    "Find WASHINGTON near WASHINGTON that accepts SCHOOL and has a SCHOOL.",

    "Find SOMETHING near WASHINGTON that accepts WASHINGTON and has a WASHINGTON.",
    "Find SOMETHING near WASHINGTON that accepts WASHINGTON and has a SOMETHING.",
    "Find SOMETHING near WASHINGTON that accepts WASHINGTON and has a SCHOOL.",
    "Find SOMETHING near WASHINGTON that accepts SOMETHING and has a WASHINGTON.",
    "Find SOMETHING near WASHINGTON that accepts SOMETHING and has a SOMETHING.",
    "Find SOMETHING near WASHINGTON that accepts SOMETHING and has a SCHOOL.",
    "Find SOMETHING near WASHINGTON that accepts SCHOOL and has a WASHINGTON.",
    "Find SOMETHING near WASHINGTON that accepts SCHOOL and has a SOMETHING.",
    "Find SOMETHING near WASHINGTON that accepts SCHOOL and has a SCHOOL.",

    "Find SCHOOL near WASHINGTON that accepts WASHINGTON and has a WASHINGTON.",
    "Find SCHOOL near WASHINGTON that accepts WASHINGTON and has a SOMETHING.",
    "Find SCHOOL near WASHINGTON that accepts WASHINGTON and has a SCHOOL.",
    "Find SCHOOL near WASHINGTON that accepts SOMETHING and has a WASHINGTON.",
    "Find SCHOOL near WASHINGTON that accepts SOMETHING and has a SOMETHING.",
    "Find SCHOOL near WASHINGTON that accepts SOMETHING and has a SCHOOL.",
    "Find SCHOOL near WASHINGTON that accepts SCHOOL and has a WASHINGTON.",
    "Find SCHOOL near WASHINGTON that accepts SCHOOL and has a SOMETHING.",
    "Find SCHOOL near WASHINGTON that accepts SCHOOL and has a SCHOOL.",

    "Find MONALISA near WASHINGTON that accepts WASHINGTON and has a WASHINGTON.",
    "Find MONALISA near WASHINGTON that accepts WASHINGTON and has a SOMETHING.",
    "Find MONALISA near WASHINGTON that accepts WASHINGTON and has a SCHOOL.",
    "Find MONALISA near WASHINGTON that accepts SOMETHING and has a WASHINGTON.",
    "Find MONALISA near WASHINGTON that accepts SOMETHING and has a SOMETHING.",
    "Find MONALISA near WASHINGTON that accepts SOMETHING and has a SCHOOL.",
    "Find MONALISA near WASHINGTON that accepts SCHOOL and has a WASHINGTON.",
    "Find MONALISA near WASHINGTON that accepts SCHOOL and has a SOMETHING.",
    "Find MONALISA near WASHINGTON that accepts SCHOOL and has a SCHOOL.",

    "Find NOTEBOOK near WASHINGTON that accepts WASHINGTON and has a WASHINGTON.",
    "Find NOTEBOOK near WASHINGTON that accepts WASHINGTON and has a SOMETHING.",
    "Find NOTEBOOK near WASHINGTON that accepts WASHINGTON and has a SCHOOL.",
    "Find NOTEBOOK near WASHINGTON that accepts SOMETHING and has a WASHINGTON.",
    "Find NOTEBOOK near WASHINGTON that accepts SOMETHING and has a SOMETHING.",
    "Find NOTEBOOK near WASHINGTON that accepts SOMETHING and has a SCHOOL.",
    "Find NOTEBOOK near WASHINGTON that accepts SCHOOL and has a WASHINGTON.",
    "Find NOTEBOOK near WASHINGTON that accepts SCHOOL and has a SOMETHING.",
    "Find NOTEBOOK near WASHINGTON that accepts SCHOOL and has a SCHOOL."
]

four_class_id = [
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7
]
message_total = [one,two,three,four]
message_embeddings_total=[]
embeded_class_id=[one_class_id,two_class_id,three_class_id, four_class_id]


for i in range(len(message_total)):
    print(len(message_total[i]))
    print(len(embeded_class_id[i]))
    assert len(message_total[i]) !=embeded_class_id[i]

# script_total = [one_script, two_script, three_script, four_script]
# class_total = [
#     "[SEARCH FROM:PH0 WHERE:PH1 WHEN:PH2]",
#     "[SEARCH FROM:PH0 WHERE:PH1]",
#     "[SEARCH FROM:PH0 WHERE:[SEARCH GEOCODE WHERE:PH1]]",
#     "[SEARCH FROM:PH0 WHERE:PH1 WITH:PH2]",
#     "[SEARCH ONE FROM:PH0 WHERE:PH1]",
#     "[SEARCH ONE FROM:PH0 WHERE:PH1 RANGE:Distance WITH:[SORT PRICE ASC]]",
#     "[SEARCH ONE FROM:PH0 WHERE:PH1 WITH:PH2]",
#     "[SEARCH ONE FROM:PH0 WITH:PH1 WITH:PH2]",
#     "[ROUTE TO:[SEARCH KEYWORD:PH0]]",
#     "[ROUTE INFO:PH0]",
#     "[ROUTE PH0]",
#     "[ROUTE PH0 USE:[SEARCH LINKS:PH1]]",
#     "[MODE PH0 OVERVIEW]",
#     "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:PH0]]]",
#     "[MODE DRIVERANGE]",
#     "[MODE DRIVERANGE TO:[SEARCH KEYWORD:PH0 FROM:PH1 WHEN:PH2] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
#     "[MODE TRAFFIC [SEARCH FROM:PH0 WHERE:[WEARCH KEYWORD:PH1]] WITH:[VOICERESPONSE TEMPLATE:*]]",
#     "[MODE SPEEDCAMERA WHERE:PH0 WITH:[VOICERESPONSE TEMPLATE:*]]",
#     "[MODE WHEATHERFORECAST WHERE:[SEARCH KEYWORD:PH0] WHEN:PH1]"
# ]

class_total = [
    "[SEARCH FROM:PH0 WHERE:HERE WHEN:Time]",
    "[SEARCH FROM:PH0 WHERE:PH1]",
    "[SEARCH FROM:PH0 WHERE:[SEARCH GEOCODE WHERE:PH1 and PH2]]",
    "[SEARCH FROM:PH0 WHERE:NEARBY WITH:PH1]",
    "[SEARCH ONE FROM:PH0 WHERE:PH1]",
    "[SEARCH ONE FROM:PH0 WHERE:PH1 RANGE:Distance WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:PH0 WHERE:PH1 WITH:PH2]",
    "[SEARCH ONE FROM:PH0 WITH:PH2 WITH:PH3]",
    "[ROUTE TO:[SEARCH KEYWORD:PH0]]",
    "[ROUTE INFO:PH0]",
    "[ROUTE PH0]",
    "[ROUTE PH0 USE:[SEARCH LINKS:PH1]]",
    "[MODE PH0 OVERVIEW]",
    "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:PH0]]]",
    "[MODE DRIVERANGE]",
    "[MODE DRIVERANGE TO:[SEARCH KEYWORD:PH0 FROM:PH1 WHEN:Time] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
    "[MODE TRAFFIC [SEARCH FROM:PH0 WHERE:[WEARCH KEYWORD:PH1]] WITH:[VOICERESPONSE TEMPLATE:*]]",
    "[MODE SPEEDCAMERA WHERE:PH0 WITH:[VOICERESPONSE TEMPLATE:*]]",
    "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:PH0] WHEN:Time]"
]


google_entity_type ={0:'UNKNOWN', 1:'PERSON', 2:'LOCATION', 3:'ORGANIZATION', 4:'EVENT', 5:'WORK_OF_ART', 6:'CONSUMER_GOOD', 7:'OTHER'}
entity_type ={0:'X', 1:'DAVID', 2:'WASHINGTON', 3:'SCHOOL', 4:'MEETTING', 5:'MONALISA', 6:'NOTEBOOK', 7:'SOMETHING'}
i_entity_type ={'X':0, 'DAVID':1, 'WASHINGTON':2, 'SCHOOL':3, 'MEETTING':4, 'MONALISA':5, 'NOTEBOOK':6, 'SOMETHING':7}
# Times = ['tomorrow', 'the day after tomorrow', 'next week', 'afternoon', 'morning', 'evening', 'dawn', 'midnight', 'noon']
Times = ['tomorrow', 'the day after tomorrow', 'next week', 'afternoon', 'morning', 'evening', 'dawn', 'midnight', 'noon']
Distance = ['50 meters', '100 meters', '500 meters', '1000 meters', '2000 meters']
line_time=''
line_distance=''

#
# entity_type ={0:'X', 1:'DAVID', 2:'WASHINGTON', 3:'SCHOOL', 4:'MEETTING', 5:'MONALISA', 6:'NOTEBOOK', 7:'SOMETHING'}
# i_entity_type ={'X':0, 'DAVID':1, 'WASHINGTON':2, 'SCHOOL':3, 'MEETTING':4, 'MONALISA':5, 'NOTEBOOK':6, 'SOMETHING':7}
# distance = ['meters']
def check_order(sentence, nouns):
    position = {}
    result_dic={}
    for noun in nouns:
        position[noun] = sentence.find(noun)
    result = sorted(position, key=position.__getitem__)
    print('result : ', result)
    for num,key in enumerate(result):
        result_dic.update({key : "PH"+str(num) })
    return result_dic


def find_and_change_entity(input_sentence):
    entity_counter = [0, 0, 0, 0, 0, 0, 0, 0]
    #Preprocess: Find time domain words.
    # times = ['tomorrow', 'the day after tomorrow', 'next week', 'afternoon', 'morning', 'evening', 'dawn', 'midnight', 'noon']
    times = ['tomorrow', 'the day after tomorrow', 'next week', 'afternoon', 'morning', 'evening', 'dawn', 'midnight', 'noon']
    Distance = ['50 meters', '100 meters', '500 meters', '1000 meters', '2000 meters']
    Line_time=''
    Line_distance=''
    replaced_sentence = input_sentence

    client = language.LanguageServiceClient()
    document = types.Document(
        content=input_sentence,
        language='en',
        type=enums.Document.Type.PLAIN_TEXT)

    entities = client.analyze_entities(document).entities

    replace_save_dict = {}
    for i in range(len(times)):
        if times[i] in replaced_sentence:
            Line_time = times[i]
            replaced_sentence = replaced_sentence.replace(times[i], 'Time')
    for i in range(len(Distance)):
        if Distance[i] in replaced_sentence:
            Line_distance = Distance[i]
            replaced_sentence = replaced_sentence.replace(Distance[i], 'Distance')

    for i in range(len(Default_where)):
        if Default_where[i] in replaced_sentence:
            Line_distance = Default_where[i]
            replaced_sentence = replaced_sentence.replace(Default_where[i], 'Distance')

    for number, entity in enumerate(entities):
        replaced_sentence = replaced_sentence.replace(entity.name, entity_type[entity.type])
        print("replaced_sentence: {}".format(replaced_sentence))
        # if entity.name not in replace_save_dict:
        #     replace_save_dict.update({entity.name:"PH"+str(number)})
        print("replace_save_dict: {}".format(replace_save_dict))
        print("entity_counter: {}".format(entity_counter))
        replace_save_dict.update({entity.name: "PH" + str(number)})
    replace_save_dict = check_order(input_sentence, replace_save_dict)
    return replaced_sentence, replace_save_dict, Line_time, Line_distance

def replace_to_script(input_script, replace_save_dict, Line_time, Line_distance):
    for replace_element in replace_save_dict:
        input_script = input_script.replace(replace_save_dict[replace_element], replace_element)
    if Line_time is not '':
        input_script = input_script.replace('Time', Line_time,1)
    if Line_distance is not '':
        input_script = input_script.replace('Distance', Line_distance, 1)
    replaced_script = input_script
    return replaced_script

# def main():
#     input_sentence = "Can you find me a gas station with restroom facilities nearby?"
#     skeleton_script = "[SEARCH FROM:WASHINGTON0 WHERE:NEARBY WITH:WASHINGTON1]"
#
#     time0 = time.time()
#     entity_changed_sentence, replace_saved_dict = find_and_change_entity(input_sentence)
#     time1 = time.time()
#     result = replace_to_script(skeleton_script, replace_saved_dict)
#     time2 = time.time()
#     print("run time0: {}".format(time1 - time0))
#     print("run time1: {}".format(time2 - time1))
#     print("total run time: {}".format(time2 - time0))
#     print("replaced_saved_dict: {}".format(replace_saved_dict))
#     print("entity_saved_sentence: {}".format(entity_changed_sentence))
#     print("skeleton_script: {}".format(skeleton_script))
#     print("result: {}".format(result))
#
# if __name__ == '__main__':
#     main()


# session.run([tf.initialize_all_variables(), tf.local_variables_initializer()])
t_sentences, t_scripts, class_ID = load_data(Data_path)

print("start embed")
# Import the Universal Sentence Encoder's TF Hub module

# w=open("Result_.txt",'a')
Correct_ID_num=0
Correct_nav_num=0
Correct_all = 0
module_url = "https://tfhub.dev/google/universal-sentence-encoder/1"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/1", "https://tfhub.dev/google/universal-sentence-encoder-large/1"]
embed = hub.Module(module_url)
print("end")
messages = tf.placeholder(dtype=tf.string, shape=[None])
embedding = embed(messages)
session = tf.Session()
session.run([tf.global_variables_initializer(), tf.tables_initializer()])
print("end session")
for test_enum, Input_data in enumerate(zip(t_sentences,t_scripts,class_ID)):
    time0 = time.time()
    lines=Input_data[0]
    List_script = []
    num = []
    embed_script = []
    entity_changed_sentence, replace_save_dict, line_time, line_distance = find_and_change_entity(lines)
    # light_module = True
    # client = language.LanguageServiceClient()
    # document = types.Document(
    #     content=lines,
    #     language='en',
    #     type=enums.Document.Type.PLAIN_TEXT)
    # # document = types.Document(content=lines,language='en',type=enums.Document.Type.PLAIN_TEXT)
    # entities = client.analyze_entities(document).entities
    #
    # result = Input_data[0]
    #
    # for entity in entities:
    #     result = result.replace(entity.name, entity_type[entity.type])

    print("\n\ninput : {}".format(Input_data))
    print("Replace nouns: {}\n".format(entity_changed_sentence))

    time1 = time.time()
    tf.logging.set_verbosity(tf.logging.ERROR)


    if not os.path.exists("./profile2.bin"):
        print("There is no profile2.bin. Making profile2.bin")
        for message in message_total:
            message_embeddings = session.run(embed(message))
            message_embeddings_total.append(message_embeddings)
        with open('profile2.bin', 'wb') as f:
            pickle.dump(message_embeddings_total, f)
        print("Finish make profile!")
    else:
        print("profile2.bin exists")
        with open('./profile2.bin', 'rb') as f:
            message_embeddings_total = pickle.load(f)

    print("\n\ncompare with : "+ str(len(message_total[0]))+", "+str(len(message_total[1]))+", "+str(len(message_total[2])))
    print("compare with : "+ str(len(message_embeddings_total[0]))+", "+str(len(message_embeddings_total[1]))+", "+str(len(message_embeddings_total[2])))


    time2 = time.time()
    test_message_embeddings = session.run(embedding, feed_dict={messages: [entity_changed_sentence]})
    time3 = time.time()

    minimum = 100
    minimum_index = 0
    entity_num = len(replace_save_dict)
    if entity_num==0:
        entity_num=1
    print('entity num : ', entity_num)

    for i, message_embedding in enumerate(message_embeddings_total[entity_num-1]):
        error = rmse(np.array(message_embedding), np.array(test_message_embeddings))
        if minimum > error:
          minimum = error
          minimum_index = i

    print("Minimum RMSE value: {}".format(minimum))
    # print("Most similar script: {}".format(script_total[entity_num - 1][num[minimum_index]]))
    print("Estimation: {}".format(minimum_index))
    print("number : ", embeded_class_id[entity_num - 1][minimum_index])
    print("Most similar script: {}".format(class_total[embeded_class_id[entity_num - 1][minimum_index]]))
    # print("Estimation: {}".format(minimum_index))
    #print("Answer: {}\n".format(test_label))
    # result2 = script_total[entity_num - 1][minimum_index]  # query
    result2 = class_total[embeded_class_id[entity_num - 1][minimum_index]]  # query
    Dict_entitis={}
    K=[] #Keys = Type
    V=[] #Values = Names
    google=[]
    # for entity in entities:
    #     # Dict_entitis[entity.name]=entity_type[entity.type]
    #     google.append(google_entity_type[entity.type])
    #     K.append(entity_type[entity.type])
    #     V.append(entity.name)


    # print("Dict_entitis : ", Dict_entitis)
    # print("Entities : ", google)
    # print("Keys : ", K)
    # print("Values : ", V)
    print("entity_number : ", entity_num)

    result3 = replace_to_script(result2, replace_save_dict, line_time, line_distance)
    result3 = filling_default(result3)
    print("input: {}".format(Input_data[0]))
    print("Replace nouns: {}".format(replace_save_dict))
    print("Selected Sentence: {}".format(message_total[entity_num-1][minimum_index]))
    print("Query: {}".format(result2))
    print("classID : ", embeded_class_id[entity_num - 1][minimum_index])
    print("answer : ", Input_data[2])
    print("result3 : ", result3.upper().replace(" ",""))
    print("Navscript : ", Input_data[1].upper().replace(" ",""))

    if str(embeded_class_id[entity_num-1][minimum_index]) != str(Input_data[2]):
        print("Wrong")
        if str(result3.upper().replace(" ","")) != str(Input_data[1].upper().replace(" ","")):
            navscript="X"
            Class = "X"
            print("Nav Wrong")

        else:
            navscript = "O"
            Class = "X"
            print("Nav Correct")
            Correct_nav_num = Correct_nav_num+1

    else:
        print("Correct")
        if str(result3.upper().replace(" ","")) != str(Input_data[1].upper().replace(" ","")):
            navscript = "X"
            Class = "O"
            print("Nav Wrong")
            Correct_ID_num = Correct_ID_num +1
        else:
            navscript = "O"
            Class = "O"
            print("Nav Correct")
            Correct_ID_num = Correct_ID_num + 1
            Correct_nav_num = Correct_nav_num + 1
            Correct_all = Correct_all + 1
    sentence = Input_data[0]
    script = Input_data[1]
    classID = embeded_class_id[entity_num - 1][minimum_index]
    answer = Input_data[2]

    if navscript =="X":
        w.write(str(test_enum) +"||"+ str(answer) + "||" + str(script) + "||" + str(result3) + "||" + str(classID)+"||"+ str(Class) + "||" + str(navscript)  + "\n")
        w2.write(str(test_enum) +"||"+ str(sentence) + "||" + str(entity_changed_sentence) + "||" + str(replace_save_dict) + "||" + str(message_total[entity_num-1][minimum_index]) +  "||" + str(result2) + "||" + str(result3)+ "||" + str(script)  +"||"+ str(Class) + "||" + str(navscript)  + "\n")
    else:
        w.write(str(test_enum) + "||" + str(answer) + "||" + str(script) + "||" + str(result3) + "||" + str(classID) + "||" + str(Class) + "||" + str(navscript) + "\n")
        w2.write(str(test_enum) + "||" + str(sentence) + "||" + str(entity_changed_sentence) + "||" + str(replace_save_dict) + "||" + str(message_total[entity_num - 1][minimum_index]) + "||" + str(result2) + "||" + str(result3) + "||" + str(script) + "||" + str(Class) + "||" + str(navscript) + "\n")
    w3.write(str(test_enum)+',' + str(test_message_embeddings) + ',' + str(answer) + "\n")
    time4 = time.time()

    print('time0 = {}'.format(time1 - time0))
    print('time1 = {}'.format(time2 - time1))
    print('time2 = {}'.format(time3 - time2))
    print('time3 = {}'.format(time4 - time3))
    print('total time = {}'.format(time4 - time0))
    print('prepare time = {}'.format(time0 - time_end))
    time_end = time.time()

print("Correct_ID : "+str(Correct_ID_num))
print("Correct_nav_num : "+str(Correct_nav_num))
print("Correct_all :  "+str(Correct_all))
print("Accuracy ID : "+str(Correct_ID_num/len(t_sentences)*100))
print("Accuracy nav : "+str(Correct_nav_num/len(t_sentences)*100))
print("Accuracy all: "+str(Correct_all/len(t_sentences)*100))

w.write("Correct_ID : "+str(Correct_ID_num))
w.write("Correct_nav_num : "+str(Correct_nav_num))
w.write("Correct_all :  "+str(Correct_all))
w.write("Accuracy ID : "+str(Correct_ID_num/len(t_sentences)*100))
w.write("Accuracy nav : "+str(Correct_nav_num/len(t_sentences)*100))
w.write("Accuracy all: "+str(Correct_all/len(t_sentences)*100))
w.close()
