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

Data_path='dataset/Test1_500.txt'
# w=open("Result1_500.txt",'a')
time_end = time.time()
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())

if len(sys.argv) > 1:
    lines = sys.argv[1]
else:
    lines =  "What's the weather forecast for this afternoon?"

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
    "[SEARCH FROM:PH0 WHERE:HERE WHEN:PH2]",
    "[SEARCH FROM:PH0 WHERE:PH1]",
    "[SEARCH FROM:PH0 WHERE:[SEARCH GEOCODE WHERE:PH1]]",
    "[SEARCH FROM:PH0 WHERE:NEARBY WITH:PH2]",
    "[SEARCH ONE FROM:PH0 WHERE:PH1]",
    "[SEARCH ONE FROM:PH0 WHERE:PH1 RANGE:Distance WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:PH0 WHERE:PH1 WITH:PH2]",
    "[SEARCH ONE FROM:PH0 WITH:PH1 WITH:PH2]",
    "[ROUTE TO:[SEARCH KEYWORD:PH0]]",
    "[ROUTE INFO:PH0]",
    "[ROUTE PH0]",
    "[ROUTE PH0 USE:[SEARCH LINKS:PH1]]",
    "[MODE PH0 OVERVIEW]",
    "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:PH0]]]",
    "[MODE DRIVERANGE]",
    "[MODE DRIVERANGE TO:[SEARCH KEYWORD:PH0 FROM:PH1 WHEN:PH2] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
    "[MODE TRAFFIC [SEARCH FROM:PH0 WHERE:[WEARCH KEYWORD:PH1]] WITH:[VOICERESPONSE TEMPLATE:*]]",
    "[MODE SPEEDCAMERA WHERE:PH0 WITH:[VOICERESPONSE TEMPLATE:*]]",
    "[MODE WHEATHERFORECAST WHERE:[SEARCH KEYWORD:PH0] WHEN:PH1]"
]


google_entity_type ={0:'UNKNOWN', 1:'PERSON', 2:'LOCATION', 3:'ORGANIZATION', 4:'EVENT', 5:'WORK_OF_ART', 6:'CONSUMER_GOOD', 7:'OTHER'}
entity_type ={0:'X', 1:'DAVID', 2:'WASHINGTON', 3:'SCHOOL', 4:'MEETTING', 5:'MONALISA', 6:'NOTEBOOK', 7:'SOMETHING'}
i_entity_type ={'X':0, 'DAVID':1, 'WASHINGTON':2, 'SCHOOL':3, 'MEETTING':4, 'MONALISA':5, 'NOTEBOOK':6, 'SOMETHING':7}
# Times = ['tomorrow', 'the day after tomorrow', 'next week', 'afternoon', 'morning', 'evening', 'dawn', 'midnight', 'noon']
Times = ['tomorrow', 'the day after tomorrow', 'next week', 'this afternoon', 'this morning', 'this evening', 'this dawn', 'this midnight', 'this noon']
Distance = ['50 meters', '100 meters', '500 meters', '1000 meters', '2000 meters']
line_time=''
line_distance=''

#
# entity_type ={0:'X', 1:'DAVID', 2:'WASHINGTON', 3:'SCHOOL', 4:'MEETTING', 5:'MONALISA', 6:'NOTEBOOK', 7:'SOMETHING'}
# i_entity_type ={'X':0, 'DAVID':1, 'WASHINGTON':2, 'SCHOOL':3, 'MEETTING':4, 'MONALISA':5, 'NOTEBOOK':6, 'SOMETHING':7}
# distance = ['meters']

def find_and_change_entity(input_sentence):
    entity_counter = [0, 0, 0, 0, 0, 0, 0, 0]
    #Preprocess: Find time domain words.
    # times = ['tomorrow', 'the day after tomorrow', 'next week', 'afternoon', 'morning', 'evening', 'dawn', 'midnight', 'noon']
    times = ['tomorrow', 'the day after tomorrow', 'next week', 'this afternoon', 'this morning', 'this evening', 'this dawn', 'this midnight', 'this noon']
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
    print('entities : ', entities)
    replace_save_dict = {}
    for i in range(len(times)):
        if times[i] in replaced_sentence:
            Line_time = times[i]
            replaced_sentence = replaced_sentence.replace(times[i], 'Time')
    for i in range(len(Distance)):
        if Distance[i] in replaced_sentence:
            Line_distance = Distance[i]
            replaced_sentence = replaced_sentence.replace(Distance[i], 'Distance')

    for entity in entities:
        replaced_sentence = replaced_sentence.replace(entity.name, entity_type[entity.type] + str(entity_counter[entity.type]))
        if entity.name not in replace_save_dict:
            replace_save_dict.update({entity.name:entity_type[entity.type]+str(entity_counter[entity.type])})
        entity_counter[entity.type] += 1

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


# session.run([tf.initialize_all_variables(), tf.local_variables_initializer()])
t_sentences, t_scripts, class_ID = load_data(Data_path)

print("start embed")
# Import the Universal Sentence Encoder's TF Hub module
module_url = "https://tfhub.dev/google/universal-sentence-encoder/1"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/1", "https://tfhub.dev/google/universal-sentence-encoder-large/1"]
embed = hub.Module(module_url)
print("end")
messages = tf.placeholder(dtype=tf.string, shape=[None])
embedding = embed(messages)
session = tf.Session()
session.run([tf.global_variables_initializer(), tf.tables_initializer()])
print("end session")
time0 = time.time()
entity_changed_sentence, replace_save_dict, line_time, line_distance = find_and_change_entity(lines)

print("\n\ninput : {}".format(lines))
print("Replace nouns: {}\n\n".format(entity_changed_sentence))
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
List_script = []
num = []
embed_script = []

for value in replace_save_dict.values():
    if value == list(replace_save_dict.values())[0]:
        for number, message in enumerate(message_total[entity_num - 1]):
            if value in message:
                List_script.append(message)
                num.append(number)
    else:
        temp_message = []
        temp_num = []
        for i in range(len(List_script)):
            if value in List_script[i]:
                temp_message.append(List_script[i])
                temp_num.append(num[i])
        List_script = temp_message
        num = temp_num

for i in range(len(num)):
    embed_script.append(message_embeddings_total[entity_num - 1][num[i]])

for i, message_embedding in enumerate(embed_script):
    error = rmse(np.array(message_embedding), np.array(test_message_embeddings))
    if minimum > error:
        minimum = error
        minimum_index = i


print("Minimum RMSE value: {}".format(minimum))
print("Estimation: {}".format(minimum_index))
print("number : ", embeded_class_id[entity_num - 1][minimum_index])
print("Most similar script: {}".format(class_total[embeded_class_id[entity_num - 1][minimum_index]]))
result2 = class_total[embeded_class_id[entity_num - 1][minimum_index]]  # query
print("entity_number : ", entity_num)

result3 = replace_to_script(result2, replace_save_dict, line_time, line_distance)
result3 = filling_default(result3)
print("input: {}".format(lines))
print("Replace nouns: {}".format(replace_save_dict))
print("Selected Sentence: {}".format(message_total[entity_num-1][minimum_index]))
print("Query: {}".format(result2))
print("classID : ", embeded_class_id[entity_num - 1][minimum_index])
print("Predict : ", result3.upper())



# print("Minimum RMSE value: {}".format(minimum))
# print("Estimation: {}".format(minimum_index))
# print("num: {}".format(num))
# print("replace_save_dict : {}".format(replace_save_dict))
# print("Most similar script: {}".format(script_total[entity_num - 1][num[minimum_index]]))
#
# #print("Answer: {}\n".format(test_label))
# result2 = script_total[entity_num - 1][num[minimum_index]]  # query
#
# print("entity_number : ", entity_num)
#
# result3 = replace_to_script(result2, replace_save_dict, line_time, line_distance)
# print("input: {}".format(lines))
# print("Replace nouns: {}".format(replace_save_dict))
# print("Selected Sentence: {}".format(message_total[entity_num-1][num[minimum_index]]))
# print("Query: {}".format(result2))
# print("classID : ", embeded_class_id[entity_num - 1][num[minimum_index]])
# print("Predict : ", result3)

time4 = time.time()

print('time0 = {}'.format(time1 - time0))
print('time1 = {}'.format(time2 - time1))
print('time2 = {}'.format(time3 - time2))
print('time3 = {}'.format(time4 - time3))
print('total time = {}'.format(time4 - time0))
time_end = time.time()






