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

one =[
    "What's the WASHINGTON0 for Time?",
    "What's the SCHOOL0 for Time?",
    "What's the MEETING0 for Time?",
    "What's the MONALISA0 for Time?",
    "What's the NOTEBOOK0 for Time?",
    "What's the SOMETHING0 for Time?",

    "Navigate to WASHINGTON0.",
    "Navigate to SOMETHING0.",
    "Navigate to SCHOOL0.",
    "Navigate to MONALISA0.",
    "Navigate to DAVID0.",

    "What's my WASHINGTON0 to destination?",
    "What's my SOMETHING0 to destination?",
    "What's my SCHOOL0 to destination?",

    "Show me alternative WASHINGTON0.",
    "Show me alternative SOMETHING0.",
    "Show me alternative SCHOOL0.",
    "Show me alternative MONALISA0.",

    "Reroute using WASHINGTON0.",
    "Reroute using SOMETHING0.",
    "Reroute using SCHOOL0.",
    "Reroute using MONALISA0.",

    "Drive to WASHINGTON0.",
    "Drive to SOMETHING0.",
    "Drive to SCHOOL0.",
    "Drive to MONALISA0.",

    "What's my WASHINGTON0?",
    "What's my SOMETHING0?",
    "What's my SCHOOL0?",
    "What's my MONALISA0?",

    "Can I make Time's 10am WASHINGTON0 without recharging?",
    "Can I make Time's 10am SOMETHING0 without recharging?",
    "Can I make Time's 10am SCHOOL0 without recharging?",
    "Can I make Time's 10am MEETING0 without recharging?",

    "Will it rain Time in WASHINGTON0?",
    "Will it rain Time in SOMETHING0?",
    "Will it rain Time in SCHOOL0?",
    "Will it rain Time in MEETING0?",

    "How long can I go?",
    "How far I can go?",
    "How much longer can I go?"
]
one_script = [
    "[SEARCH FROM:WASHINGTON0  WHERE:HERE WHEN:Time]",
    "[SEARCH FROM:SCHOOL0  WHERE:HERE WHEN:Time]",
    "[SEARCH FROM:MEETING0  WHERE:HERE WHEN:Time]",
    "[SEARCH FROM:MONALISA0  WHERE:HERE WHEN:Time]",
    "[SEARCH FROM:NOTEBOOK0  WHERE:HERE WHEN:Time]",
    "[SEARCH FROM:SOMETHING0  WHERE:HERE WHEN:Time]",

    "[ROUTE TO:[SEARCH KEYWORD:WASHINGTON0]]",
    "[ROUTE TO:[SEARCH KEYWORD:SOMETHING0]]",
    "[ROUTE TO:[SEARCH KEYWORD:SCHOOL0]]",
    "[ROUTE TO:[SEARCH KEYWORD:MONALISA0]]",
    "[ROUTE TO:[SEARCH KEYWORD:DAVID0]]",

    "[ROUTE INFO:WASHINGTON0]",
    "[ROUTE INFO:SOMETHING0]",
    "[ROUTE INFO:SCHOOL0]",

    "[ROUTE WASHINGTON0]",
    "[ROUTE SOMETHING0]",
    "[ROUTE SCHOOL0]",
    "[ROUTE MONALISA0]",

    "[ROUTE ALTROUTE USE:[SEARCH LINKS:WASHINGTON0]]",
    "[ROUTE ALTROUTE USE:[SEARCH LINKS:SOMETHING0]]",
    "[ROUTE ALTROUTE USE:[SEARCH LINKS:SCHOOL0]]",
    "[ROUTE ALTROUTE USE:[SEARCH LINKS:MONALISA0]]",

    "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:WASHINGTON0]]]",
    "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:SOMETHING0]]]",
    "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:SCHOOL0]]]",
    "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:MONALISA0]]",

    "[MODE WASHINGTON0]",
    "[MODE SOMETHING0]",
    "[MODE SCHOOL0]",
    "[MODE MONALISA0]",

    "[MODE DRIVERANGE TO:[SEARCH KEYWORD:WASHINGTON0 FROM:SCHEDULE WHEN:Time] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
    "[MODE DRIVERANGE TO:[SEARCH KEYWORD:SOMETHING0 FROM:SCHEDULE WHEN:Time] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
    "[MODE DRIVERANGE TO:[SEARCH KEYWORD:SCHOOL0 FROM:SCHEDULE WHEN:Time] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
    "[MODE DRIVERANGE TO:[SEARCH KEYWORD:MEETING0 FROM:SCHEDULE WHEN:Time] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",

    "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:WASHINGTON0] WHEN:Time]",
    "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:SOMETHING0] WHEN:Time]",
    "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:SCHOOL0] WHEN:Time]",
    "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:MEETING0] WHEN:Time]",

    "[MODE DRIVERANGE]",
    "[MODE DRIVERANGE]",
    "[MODE DRIVERANGE]"


]
one_class_id=[
    0,0,0,0,0,0,
    8,8,8,8,8,
    15,15,15,
    16,16,16,16,
    9,9,9,9,
    10,10,10,10,
    18,18,18,18,
    11,11,11,11,
    14,14,14,14,
    18,18,18
]
two = [
    "What's the WASHINGTON0 like on my WASHINGTON1?",
    "What's the WASHINGTON0 like on my SOMETHING0?",
    "What's the SOMETHING0 like on my WASHINGTON0?",
    "What's the SOMETHING0 like on my SOMETHING1?",

    "Can you find me a WASHINGTON0 with WASHINGTON1 nearby?",
    "Can you find me a WASHINGTON0 with SOMETHING0 nearby?",
    "Can you find me a WASHINGTON0 with SCHOOL0 nearby?",
    "Can you find me a SOMETHING0 with WASHINGTON0 nearby?",
    "Can you find me a SOMETHING0 with SOMETHING1 nearby?",
    "Can you find me a SOMETHING0 with SCHOOL0 nearby?",
    "Can you find me a SCHOOL0 with WASHINGTON0 nearby?",
    "Can you find me a SCHOOL0 with SOMETHING0 nearby?",
    "Can you find me a SCHOOL0 with SCHOOL1 nearby?",

    "Find a WASHINGTON0 along WASHINGTON1.",
    "Find a WASHINGTON0 along SOMETHING0.",
    "Find a WASHINGTON0 along SCHOOL0.",
    "Find a SOMETHING0 along WASHINGTON0.",
    "Find a SOMETHING0 along SOMETHING1.",
    "Find a SOMETHING0 along SCHOOL0.",
    "Find a SCHOOL0 along WASHINGTON0.",
    "Find a SCHOOL0 along SOMETHING0.",
    "Find a SCHOOL0 along SCHOOL1.",

    "Find the cheapest WASHINGTON0 within Distance of my WASHINGTON1.",
    "Find the cheapest WASHINGTON0 within Distance of my SOMETHING0.",
    "Find the cheapest WASHINGTON0 within Distance of my SCHOOL0.",
    "Find the cheapest SOMETHING0 within Distance of my WASHINGTON0.",
    "Find the cheapest SOMETHING0 within Distance of my SOMETHING1.",
    "Find the cheapest SOMETHING0 within Distance of my SCHOOL0.",
    "Find the cheapest SCHOOL0 within Distance of my WASHINGTON0.",
    "Find the cheapest SCHOOL0 within Distance of my SOMETHING0.",
    "Find the cheapest SCHOOL0 within Distance of my SCHOOL1.",

    # "What's WASHINGTON like on the WASHINGTON?",
    # "What's WASHINGTON like on the SOMETHING?",
    # "What's WASHINGTON like on the SCHOOL?",
    # "What's WASHINGTON like on the MONALISA?",
    #
    # "What's SOMETHING like on the WASHINGTON?",
    # "What's SOMETHING like on the SOMETHING?",
    # "What's SOMETHING like on the SCHOOL?",
    # "What's SOMETHING like on the MONALISA?",
    #
    # "What's SCHOOL like on the WASHINGTON?",
    # "What's SCHOOL like on the SOMETHING?",
    # "What's SCHOOL like on the SCHOOL?",
    # "What's SCHOOL like on the MONALISA?",
    #
    # "What's MONALISA like on the WASHINGTON?",
    # "What's MONALISA like on the SOMETHING?",
    # "What's MONALISA like on the SCHOOL?",
    # "What's MONALISA like on the MONALISA?",

    "Are there any WASHINGTON0 on my WASHINGTON1?",
    "Are there any WASHINGTON0 on my SOMETHING0?",
    "Are there any WASHINGTON0 on my SCHOOL0?",
    "Are there any WASHINGTON0 on my MONALISA0?",
    "Are there any WASHINGTON0 on my NOTEBOOK0?",

    "Are there any SOMETHING0 on my WASHINGTON0?",
    "Are there any SOMETHING0 on my SOMETHING1?",
    "Are there any SOMETHING0 on my SCHOOL0?",
    "Are there any SOMETHING0 on my MONALISA0?",
    "Are there any SOMETHING0 on my NOTEBOOK0?",

    "Are there any SCHOOL0 on my WASHINGTON0?",
    "Are there any SCHOOL0 on my SOMETHING0?",
    "Are there any SCHOOL0 on my SCHOOL1?",
    "Are there any SCHOOL0 on my MONALISA0?",
    "Are there any SCHOOL0 on my NOTEBOOK0?",

    "Are there any MONALISA0 on my WASHINGTON0?",
    "Are there any MONALISA0 on my SOMETHING0?",
    "Are there any MONALISA0 on my SCHOOL0?",
    "Are there any MONALISA0 on my MONALISA1?",
    "Are there any MONALISA0 on my NOTEBOOK0?",

    "Are there any NOTEBOOK0 on my WASHINGTON0?",
    "Are there any NOTEBOOK0 on my SOMETHING0?",
    "Are there any NOTEBOOK0 on my SCHOOL0?",
    "Are there any NOTEBOOK0 on my MONALISA0?",
    "Are there any NOTEBOOK0 on my NOTEBOOK1?"

]
two_script = [
    "[SEARCH FROM:WASHINGTON0  WHERE:WASHINGTON1]",
    "[SEARCH FROM:WASHINGTON0  WHERE:SOMETHING0]",
    "[SEARCH FROM:SOMETHING0 WHERE:WASHINGTON0]",
    "[SEARCH FROM:SOMETHING0 WHERE:SOMETHING1]",

    "[SEARCH FROM:WASHINGTON0 WHERE:NEARBY WITH:WASHINGTON1]",
    "[SEARCH FROM:WASHINGTON0 WHERE:NEARBY WITH:SOMETHING0]",
    "[SEARCH FROM:WASHINGTON0 WHERE:NEARBY WITH:RESTROOM0]",
    "[SEARCH FROM:SOMETHING0 WHERE:NEARBY WITH:WASHINGTON0]",
    "[SEARCH FROM:SOMETHING0 WHERE:NEARBY WITH:SOMETHING1]",
    "[SEARCH FROM:SOMETHING0 WHERE:NEARBY WITH:SCHOOL0]",
    "[SEARCH FROM:SCHOOL0 WHERE:NEARBY WITH:WASHINGTON0]",
    "[SEARCH FROM:SCHOOL0 WHERE:NEARBY WITH:SOMETHING0]",
    "[SEARCH FROM:SCHOOL0 WHERE:NEARBY WITH:SCHOOL1]",

    "[SEARCH ONE FROM:WASHINGTON0 WHERE:WASHINGTON1]",
    "[SEARCH ONE FROM:WASHINGTON0 WHERE:SOMETHING0]",
    "[SEARCH ONE FROM:WASHINGTON0 WHERE:SCHOOL0]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:WASHINGTON0]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:SOMETHING1]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:SCHOOL0]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:WASHINGTON0]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:SOMETHING0]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:SCHOOL1]",

    "[SEARCH ONE FROM:WASHINGTON0 WHERE:WASHINGTON1 RANGE:Distance WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:WASHINGTON0 WHERE:SOMETHING0 RANGE:Distance WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:WASHINGTON0 WHERE:SCHOOL0 RANGE:Distance WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:WASHINGTON0 RANGE:Distance WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:SOMETHING1 RANGE:Distance WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:SCHOOL0 RANGE:Distance WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:WASHINGTON0 RANGE:Distance WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:SOMETHING0 RANGE:Distance WITH:[SORT PRICE ASC]]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:SCHOOL1 RANGE:Distance WITH:[SORT PRICE ASC]]",

    # "[MODE TRAFFIC [SEARCH FROM:WASHINGTON WHERE:[SEARCH KEYWORD:WASHINGTON]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:WASHINGTON WHERE:[SEARCH KEYWORD:SOMETHING]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:WASHINGTON WHERE:[SEARCH KEYWORD:SCHOOL]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:WASHINGTON WHERE:[SEARCH KEYWORD:MONALISA]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    #
    # "[MODE TRAFFIC [SEARCH FROM:SOMETHING WHERE:[SEARCH KEYWORD:WASHINGTON]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:SOMETHING WHERE:[SEARCH KEYWORD:SOMETHING]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:SOMETHING WHERE:[SEARCH KEYWORD:SCHOOL]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:SOMETHING WHERE:[SEARCH KEYWORD:MONALISA]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    #
    # "[MODE TRAFFIC [SEARCH FROM:SCHOOL WHERE:[SEARCH KEYWORD:WASHINGTON]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:SCHOOL WHERE:[SEARCH KEYWORD:SOMETHING]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:SCHOOL WHERE:[SEARCH KEY0WORD:SCHOOL]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:SCHOOL WHERE:[SEARCH KEYWORD:MONALISA]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    #
    # "[MODE TRAFFIC [SEARCH FROM:MONALISA WHERE:[SEARCH KEYWORD:WASHINGTON]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:MONALISA WHERE:[SEARCH KEYWORD:SOMETHING]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:MONALISA WHERE:[SEARCH KEYWORD:SCHOOL]] WITH:[VOICERESPONSE TEMPLATE:””*]",
    # "[MODE TRAFFIC [SEARCH FROM:MONALISA WHERE:[SEARCH KEYWORD:MONALISA]] WITH:[VOICERESPONSE TEMPLATE:””*]",

    "[MODE WASHINGTON0 WHERE:WASHINGTON1 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE WASHINGTON0 WHERE:SOMETHING0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE WASHINGTON0 WHERE:SCHOOL0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE WASHINGTON0 WHERE:MONALISA0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE WASHINGTON0 WHERE:NOTEBOOK0 WITH:[VOICERESPONSE TEMPLATE:””*]]",

    "[MODE SOMETHING0 WHERE:WASHINGTON0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SOMETHING0 WHERE:SOMETHING1 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SOMETHING0 WHERE:SCHOOL0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SOMETHING0 WHERE:MONALISA0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SOMETHING0 WHERE:NOTEBOOK0 WITH:[VOICERESPONSE TEMPLATE:””*]]",

    "[MODE SCHOOL0 WHERE:WASHINGTON0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SCHOOL0 WHERE:SOMETHING0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SCHOOL0 WHERE:SCHOOL1 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SCHOOL0 WHERE:MONALISA0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE SCHOOL0 WHERE:NOTEBOOK0 WITH:[VOICERESPONSE TEMPLATE:””*]]",

    "[MODE MONALISA0 WHERE:WASHINGTON0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE MONALISA0 WHERE:SOMETHING0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE MONALISA0 WHERE:SCHOOL0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE MONALISA0 WHERE:MONALISA1 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE MONALISA0 WHERE:NOTEBOOK0 WITH:[VOICERESPONSE TEMPLATE:””*]]",

    "[MODE NOTEBOOK0 WHERE:WASHINGTON0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE NOTEBOOK0 WHERE:SOMETHING0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE NOTEBOOK0 WHERE:SCHOOL0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE NOTEBOOK0 WHERE:MONALISA0 WITH:[VOICERESPONSE TEMPLATE:””*]]",
    "[MODE NOTEBOOK0 WHERE:NOTEBOOK1 WITH:[VOICERESPONSE TEMPLATE:””*]]"
]
two_class_id=[
    1, 1, 1, 1,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5,
    13, 13, 13, 13, 13,
    13, 13, 13, 13, 13,
    13, 13, 13, 13, 13,
    13, 13, 13, 13, 13,
    13, 13, 13, 13, 13

]
three = [
    "Show me a WASHINGTON0 on WASHINGTON1 and WASHINGTON2.",
    "Show me a WASHINGTON0 on WASHINGTON1 and SOMETHING0.",
    "Show me a WASHINGTON0 on WASHINGTON1 and SCHOOL0.",
    "Show me a WASHINGTON0 on SOMETHING0 and WASHINGTON1.",
    "Show me a WASHINGTON0 on SOMETHING0 and SCHOOL0.",
    "Show me a WASHINGTON0 on SOMETHING0 and SOMETHING1.",
    "Show me a WASHINGTON0 on SCHOOL0 and WASHINGTON1.",
    "Show me a WASHINGTON0 on SCHOOL0 and SOMETHING0.",
    "Show me a WASHINGTON0 on SCHOOL0 and SCHOOL1.",

    "Show me a SOMETHING0 on WASHINGTON0 and WASHINGTON1.",
    "Show me a SOMETHING0 on WASHINGTON0 and SOMETHING1.",
    "Show me a SOMETHING0 on WASHINGTON0 and SCHOOL0.",
    "Show me a SOMETHING0 on SOMETHING1 and WASHINGTON0.",
    "Show me a SOMETHING0 on SOMETHING1 and SCHOOL0.",
    "Show me a SOMETHING0 on SOMETHING1 and SOMETHING2.",
    "Show me a SOMETHING0 on SCHOOL0 and WASHINGTON0.",
    "Show me a SOMETHING0 on SCHOOL0 and SOMETHING1.",
    "Show me a SOMETHING0 on SCHOOL0 and SCHOOL1.",

    "Show me a SCHOOL0 on WASHINGTON0 and WASHINGTON1.",
    "Show me a SCHOOL0 on WASHINGTON0 and SOMETHING0.",
    "Show me a SCHOOL0 on WASHINGTON0 and SCHOOL1.",
    "Show me a SCHOOL0 on SOMETHING0 and WASHINGTON0.",
    "Show me a SCHOOL0 on SOMETHING0 and SCHOOL1.",
    "Show me a SCHOOL0 on SOMETHING0 and SOMETHING1.",
    "Show me a SCHOOL0 on SCHOOL1 and WASHINGTON0.",
    "Show me a SCHOOL0 on SCHOOL1 and SOMETHING0.",
    "Show me a SCHOOL0 on SCHOOL1 and SCHOOL2.",

    "Okay, can you find me a WASHINGTON0 on my WASHINGTON1 that has a WASHINGTON2?",
    "Okay, can you find me a WASHINGTON0 on my WASHINGTON1 that has a SOMETHING0?",
    "Okay, can you find me a WASHINGTON0 on my WASHINGTON1 that has a SCHOOL0?",
    "Okay, can you find me a WASHINGTON0 on my SOMETHING0 that has a WASHINGTON1?",
    "Okay, can you find me a WASHINGTON0 on my SOMETHING0 that has a SOMETHING1?",
    "Okay, can you find me a WASHINGTON0 on my SOMETHING0 that has a SCHOOL0?",
    "Okay, can you find me a WASHINGTON0 on my SCHOOL0 that has a WASHINGTON1?",
    "Okay, can you find me a WASHINGTON0 on my SCHOOL0 that has a SOMETHING0?",
    "Okay, can you find me a WASHINGTON0 on my SCHOOL0 that has a SCHOOL1?",

    "Okay, can you find me a SOMETHING0 on my WASHINGTON0 that has a WASHINGTON1?",
    "Okay, can you find me a SOMETHING0 on my WASHINGTON0 that has a SOMETHING1?",
    "Okay, can you find me a SOMETHING0 on my WASHINGTON0 that has a SCHOOL0?",
    "Okay, can you find me a SOMETHING0 on my SOMETHING1 that has a WASHINGTON0?",
    "Okay, can you find me a SOMETHING0 on my SOMETHING1 that has a SOMETHING2?",
    "Okay, can you find me a SOMETHING0 on my SOMETHING1 that has a SCHOOL0?",
    "Okay, can you find me a SOMETHING0 on my SCHOOL0 that has a WASHINGTON0?",
    "Okay, can you find me a SOMETHING0 on my SCHOOL0 that has a SOMETHING1?",
    "Okay, can you find me a SOMETHING0 on my SCHOOL0 that has a SCHOOL2?",

    "Okay, can you find me a SCHOOL0 on my WASHINGTON0 that has a WASHINGTON1?",
    "Okay, can you find me a SCHOOL0 on my WASHINGTON0 that has a SOMETHING0?",
    "Okay, can you find me a SCHOOL0 on my WASHINGTON0 that has a SCHOOL1?",
    "Okay, can you find me a SCHOOL0 on my SOMETHING0 that has a WASHINGTON0?",
    "Okay, can you find me a SCHOOL0 on my SOMETHING0 that has a SOMETHING1?",
    "Okay, can you find me a SCHOOL0 on my SOMETHING0 that has a SCHOOL1?",
    "Okay, can you find me a SCHOOL0 on my SCHOOL1 that has a WASHINGTON0?",
    "Okay, can you find me a SCHOOL0 on my SCHOOL1 that has a SOMETHING0?",
    "Okay, can you find me a SCHOOL0 on my SCHOOL1 that has a SCHOOL2?",

    "Find WASHINGTON0 near destination that accepts WASHINGTON1 and has a WASHINGTON2.",
    "Find WASHINGTON0 near destination that accepts WASHINGTON1 and has a SOMETHING0.",
    "Find WASHINGTON0 near destination that accepts WASHINGTON1 and has a SCHOOL0.",
    "Find WASHINGTON0 near destination that accepts SOMETHING0 and has a WASHINGTON1.",
    "Find WASHINGTON0 near destination that accepts SOMETHING0 and has a SOMETHING1.",
    "Find WASHINGTON0 near destination that accepts SOMETHING0 and has a SCHOOL0.",
    "Find WASHINGTON0 near destination that accepts SCHOOL0 and has a WASHINGTON1.",
    "Find WASHINGTON0 near destination that accepts SCHOOL0 and has a SOMETHING0.",
    "Find WASHINGTON0 near destination that accepts SCHOOL0 and has a SCHOOL1.",

    "Find SOMETHING0 near destination that accepts WASHINGTON0 and has a WASHINGTON1.",
    "Find SOMETHING0 near destination that accepts WASHINGTON0 and has a SOMETHING1.",
    "Find SOMETHING0 near destination that accepts WASHINGTON0 and has a SCHOOL0.",
    "Find SOMETHING0 near destination that accepts SOMETHING1 and has a WASHINGTON0.",
    "Find SOMETHING0 near destination that accepts SOMETHING1 and has a SOMETHING2.",
    "Find SOMETHING0 near destination that accepts SOMETHING1 and has a SCHOOL0.",
    "Find SOMETHING0 near destination that accepts SCHOOL0 and has a WASHINGTON0.",
    "Find SOMETHING0 near destination that accepts SCHOOL0 and has a SOMETHING1.",
    "Find SOMETHING0 near destination that accepts SCHOOL0 and has a SCHOOL1.",

    "Find SCHOOL0 near destination that accepts WASHINGTON0 and has a WASHINGTON1.",
    "Find SCHOOL0 near destination that accepts WASHINGTON0 and has a SOMETHING0.",
    "Find SCHOOL0 near destination that accepts WASHINGTON0 and has a SCHOOL1.",
    "Find SCHOOL0 near destination that accepts SOMETHING0 and has a WASHINGTON0.",
    "Find SCHOOL0 near destination that accepts SOMETHING0 and has a SOMETHING1.",
    "Find SCHOOL0 near destination that accepts SOMETHING0 and has a SCHOOL1.",
    "Find SCHOOL0 near destination that accepts SCHOOL1 and has a WASHINGTON0.",
    "Find SCHOOL0 near destination that accepts SCHOOL1 and has a SOMETHING0.",
    "Find SCHOOL0 near destination that accepts SCHOOL1 and has a SCHOOL2.",

    "Find MONALISA0 near destination that accepts WASHINGTON0 and has a WASHINGTON1.",
    "Find MONALISA0 near destination that accepts WASHINGTON0 and has a SOMETHING0.",
    "Find MONALISA0 near destination that accepts WASHINGTON0 and has a SCHOOL0.",
    "Find MONALISA0 near destination that accepts SOMETHING0 and has a WASHINGTON0.",
    "Find MONALISA0 near destination that accepts SOMETHING0 and has a SOMETHING1.",
    "Find MONALISA0 near destination that accepts SOMETHING0 and has a SCHOOL0.",
    "Find MONALISA0 near destination that accepts SCHOOL0 and has a WASHINGTON0.",
    "Find MONALISA0 near destination that accepts SCHOOL0 and has a SOMETHING0.",
    "Find MONALISA0 near destination that accepts SCHOOL0 and has a SCHOOL1.",

    "Find NOTEBOOK0 near destination that accepts WASHINGTON0 and has a WASHINGTON1.",
    "Find NOTEBOOK0 near destination that accepts WASHINGTON0 and has a SOMETHING0.",
    "Find NOTEBOOK0 near destination that accepts WASHINGTON0 and has a SCHOOL0.",
    "Find NOTEBOOK0 near destination that accepts SOMETHING0 and has a WASHINGTON0.",
    "Find NOTEBOOK0 near destination that accepts SOMETHING0 and has a SOMETHING1.",
    "Find NOTEBOOK0 near destination that accepts SOMETHING0 and has a SCHOOL0.",
    "Find NOTEBOOK0 near destination that accepts SCHOOL0 and has a WASHINGTON0.",
    "Find NOTEBOOK0 near destination that accepts SCHOOL0 and has a SOMETHING0.",
    "Find NOTEBOOK0 near destination that accepts SCHOOL0 and has a SCHOOL1."
]
three_script = [
    "[SEARCH FROM:WASHINGTON0  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON1 and WASHINGTON2]]",
    "[SEARCH FROM:WASHINGTON0  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON1 and SOMETHING0]]",
    "[SEARCH FROM:WASHINGTON0  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON1 and SCHOOL0]]",
    "[SEARCH FROM:WASHINGTON0  WHERE:[SEARCH GEOCODE WHERE:SOMETHING0 and WASHINGTON1]]",
    "[SEARCH FROM:WASHINGTON0  WHERE:[SEARCH GEOCODE WHERE:SOMETHING0 and SCHOOL0]]",
    "[SEARCH FROM:WASHINGTON0  WHERE:[SEARCH GEOCODE WHERE:SOMETHING0 and SOMETHING1]]",
    "[SEARCH FROM:WASHINGTON0  WHERE:[SEARCH GEOCODE WHERE:SCHOOL0 and WASHINGTON1]]",
    "[SEARCH FROM:WASHINGTON0  WHERE:[SEARCH GEOCODE WHERE:SCHOOL0 and SOMETHING0]]",
    "[SEARCH FROM:WASHINGTON0  WHERE:[SEARCH GEOCODE WHERE:SCHOOL0 and SCHOOL1]]",

    "[SEARCH FROM:SOMETHING0  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON0 and WASHINGTON1]]",
    "[SEARCH FROM:SOMETHING0  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON0 and SOMETHING1]]",
    "[SEARCH FROM:SOMETHING0  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON0 and SCHOOL0]]",
    "[SEARCH FROM:SOMETHING0  WHERE:[SEARCH GEOCODE WHERE:SOMETHING1 and WASHINGTON0]]",
    "[SEARCH FROM:SOMETHING0  WHERE:[SEARCH GEOCODE WHERE:SOMETHING1 and SCHOOL0]]",
    "[SEARCH FROM:SOMETHING0  WHERE:[SEARCH GEOCODE WHERE:SOMETHING1 and SOMETHING2]]",
    "[SEARCH FROM:SOMETHING0  WHERE:[SEARCH GEOCODE WHERE:SCHOOL0 and WASHINGTON0]]",
    "[SEARCH FROM:SOMETHING0  WHERE:[SEARCH GEOCODE WHERE:SCHOOL0 and SOMETHING1]]",
    "[SEARCH FROM:SOMETHING0  WHERE:[SEARCH GEOCODE WHERE:SCHOOL0 and SCHOOL1]]",

    "[SEARCH FROM:SCHOOL0  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON0 and WASHINGTON1]]",
    "[SEARCH FROM:SCHOOL0  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON0 and SOMETHING0]]",
    "[SEARCH FROM:SCHOOL0  WHERE:[SEARCH GEOCODE WHERE:WASHINGTON0 and SCHOOL1]]",
    "[SEARCH FROM:SCHOOL0  WHERE:[SEARCH GEOCODE WHERE:SOMETHING0 and WASHINGTON0]]",
    "[SEARCH FROM:SCHOOL0  WHERE:[SEARCH GEOCODE WHERE:SOMETHING0 and SCHOOL1]]",
    "[SEARCH FROM:SCHOOL0  WHERE:[SEARCH GEOCODE WHERE:SOMETHING0 and SOMETHING1]]",
    "[SEARCH FROM:SCHOOL0  WHERE:[SEARCH GEOCODE WHERE:SCHOOL0 and WASHINGTON0]]",
    "[SEARCH FROM:SCHOOL0  WHERE:[SEARCH GEOCODE WHERE:SCHOOL0 and SOMETHING0]]",
    "[SEARCH FROM:SCHOOL0  WHERE:[SEARCH GEOCODE WHERE:SCHOOL0 and SCHOOL2]]",

    "[SEARCH ONE FROM:WASHINGTON0 WHERE:WASHINGTON1 WITH:WASHINGTON2]",
    "[SEARCH ONE FROM:WASHINGTON0 WHERE:WASHINGTON1 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:WASHINGTON0 WHERE:WASHINGTON1 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:WASHINGTON0 WHERE:SOMETHING0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:WASHINGTON0 WHERE:SOMETHING0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:WASHINGTON0 WHERE:SOMETHING0 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:WASHINGTON0 WHERE:SCHOOL0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:WASHINGTON0 WHERE:SCHOOL0 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:WASHINGTON0 WHERE:SCHOOL0 WITH:SCHOOL1]",

    "[SEARCH ONE FROM:SOMETHING0 WHERE:WASHINGTON0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:WASHINGTON0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:WASHINGTON0 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:SOMETHING1 WITH:WASHINGTON0]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:SOMETHING1 WITH:SOMETHING2]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:SOMETHING1 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:SCHOOL0 WITH:WASHINGTON0]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:SCHOOL0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:SOMETHING0 WHERE:SCHOOL0 WITH:SCHOOL1]",

    "[SEARCH ONE FROM:SCHOOL0 WHERE:WASHINGTON0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:WASHINGTON0 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:WASHINGTON0 WITH:SCHOOL1]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:SOMETHING0 WITH:WASHINGTON0]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:SOMETHING0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:SOMETHING0 WITH:SCHOOL1]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:SCHOOL1 WITH:WASHINGTON0]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:SCHOOL1 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:SCHOOL0 WHERE:SCHOOL1 WITH:SCHOOL2]",

    "[SEARCH ONE FROM:WASHINGTON0 WITH:WASHINGTON1 WITH:WASHINGTON2]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:WASHINGTON1 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:WASHINGTON1 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:SOMETHING0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:SOMETHING0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:SOMETHING0 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:SCHOOL0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:SCHOOL0 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:SCHOOL0 WITH:SCHOOL1]",

    "[SEARCH ONE FROM:SOMETHING0 WITH:WASHINGTON0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:WASHINGTON0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:WASHINGTON0 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:SOMETHING1 WITH:WASHINGTON0]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:SOMETHING1 WITH:SOMETHING2]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:SOMETHING1 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:SCHOOL0 WITH:WASHINGTON0]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:SCHOOL0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:SCHOOL0 WITH:SCHOOL1]",

    "[SEARCH ONE FROM:SCHOOL0 WITH:WASHINGTON0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:WASHINGTON0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:WASHINGTON0 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:SOMETHING0 WITH:WASHINGTON0]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:SOMETHING0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:SOMETHING0 WITH:SCHOOL1]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:SCHOOL1 WITH:WASHINGTON0]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:SCHOOL1 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:SCHOOL1 WITH:SCHOOL2]",

    "[SEARCH ONE FROM:MONALISA0 WITH:WASHINGTON0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:MONALISA0 WITH:WASHINGTON0 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:MONALISA0 WITH:WASHINGTON0 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:MONALISA0 WITH:SOMETHING0 WITH:WASHINGTON0]",
    "[SEARCH ONE FROM:MONALISA0 WITH:SOMETHING0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:MONALISA0 WITH:SOMETHING0 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:MONALISA0 WITH:SCHOOL0 WITH:WASHINGTON0]",
    "[SEARCH ONE FROM:MONALISA0 WITH:SCHOOL0 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:MONALISA0 WITH:SCHOOL0 WITH:SCHOOL1]",

    "[SEARCH ONE FROM:NOTEBOOK0 WITH:WASHINGTON0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:WASHINGTON0 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:WASHINGTON0 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:SOMETHING0 WITH:WASHINGTON0]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:SOMETHING0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:SOMETHING0 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:SCHOOL0 WITH:WASHINGTON0]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:SCHOOL0 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:SCHOOL0 WITH:SCHOOL1]"
]
three_class_id=[
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
    "Find WASHINGTON0 near WASHINGTON1 that accepts WASHINGTON2 and has a WASHINGTON3.",
    "Find WASHINGTON0 near WASHINGTON1 that accepts WASHINGTON2 and has a SOMETHING0.",
    "Find WASHINGTON0 near WASHINGTON1 that accepts WASHINGTON2 and has a SCHOOL0.",
    "Find WASHINGTON0 near WASHINGTON1 that accepts SOMETHING0 and has a WASHINGTON2.",
    "Find WASHINGTON0 near WASHINGTON1 that accepts SOMETHING0 and has a SOMETHING1.",
    "Find WASHINGTON0 near WASHINGTON1 that accepts SOMETHING0 and has a SCHOOL0.",
    "Find WASHINGTON0 near WASHINGTON1 that accepts SCHOOL0 and has a WASHINGTON2.",
    "Find WASHINGTON0 near WASHINGTON1 that accepts SCHOOL0 and has a SOMETHING0.",
    "Find WASHINGTON0 near WASHINGTON1 that accepts SCHOOL0 and has a SCHOOL1.",

    "Find SOMETHING0 near WASHINGTON0 that accepts WASHINGTON1 and has a WASHINGTON2.",
    "Find SOMETHING0 near WASHINGTON0 that accepts WASHINGTON1 and has a SOMETHING1.",
    "Find SOMETHING0 near WASHINGTON0 that accepts WASHINGTON1 and has a SCHOOL0.",
    "Find SOMETHING0 near WASHINGTON0 that accepts SOMETHING1 and has a WASHINGTON1.",
    "Find SOMETHING0 near WASHINGTON0 that accepts SOMETHING1 and has a SOMETHING2.",
    "Find SOMETHING0 near WASHINGTON0 that accepts SOMETHING1 and has a SCHOOL0.",
    "Find SOMETHING0 near WASHINGTON0 that accepts SCHOOL0 and has a WASHINGTON1.",
    "Find SOMETHING0 near WASHINGTON0 that accepts SCHOOL0 and has a SOMETHING1.",
    "Find SOMETHING0 near WASHINGTON0 that accepts SCHOOL0 and has a SCHOOL1.",

    "Find SCHOOL0 near WASHINGTON0 that accepts WASHINGTON1 and has a WASHINGTON2.",
    "Find SCHOOL0 near WASHINGTON0 that accepts WASHINGTON1 and has a SOMETHING0.",
    "Find SCHOOL0 near WASHINGTON0 that accepts WASHINGTON1 and has a SCHOOL1.",
    "Find SCHOOL0 near WASHINGTON0 that accepts SOMETHING0 and has a WASHINGTON1.",
    "Find SCHOOL0 near WASHINGTON0 that accepts SOMETHING0 and has a SOMETHING1.",
    "Find SCHOOL0 near WASHINGTON0 that accepts SOMETHING0 and has a SCHOOL1.",
    "Find SCHOOL0 near WASHINGTON0 that accepts SCHOOL1 and has a WASHINGTON1.",
    "Find SCHOOL0 near WASHINGTON0 that accepts SCHOOL1 and has a SOMETHING0.",
    "Find SCHOOL0 near WASHINGTON0 that accepts SCHOOL1 and has a SCHOOL2.",

    "Find MONALISA0 near WASHINGTON0 that accepts WASHINGTON1 and has a WASHINGTON2.",
    "Find MONALISA0 near WASHINGTON0 that accepts WASHINGTON1 and has a SOMETHING0.",
    "Find MONALISA0 near WASHINGTON0 that accepts WASHINGTON1 and has a SCHOOL0.",
    "Find MONALISA0 near WASHINGTON0 that accepts SOMETHING0 and has a WASHINGTON1.",
    "Find MONALISA0 near WASHINGTON0 that accepts SOMETHING0 and has a SOMETHING1.",
    "Find MONALISA0 near WASHINGTON0 that accepts SOMETHING0 and has a SCHOOL0.",
    "Find MONALISA0 near WASHINGTON0 that accepts SCHOOL0 and has a WASHINGTON1.",
    "Find MONALISA0 near WASHINGTON0 that accepts SCHOOL0 and has a SOMETHING0.",
    "Find MONALISA0 near WASHINGTON0 that accepts SCHOOL0 and has a SCHOOL1.",

    "Find NOTEBOOK0 near WASHINGTON0 that accepts WASHINGTON1 and has a WASHINGTON2.",
    "Find NOTEBOOK0 near WASHINGTON0 that accepts WASHINGTON1 and has a SOMETHING0.",
    "Find NOTEBOOK0 near WASHINGTON0 that accepts WASHINGTON1 and has a SCHOOL0.",
    "Find NOTEBOOK0 near WASHINGTON0 that accepts SOMETHING0 and has a WASHINGTON1.",
    "Find NOTEBOOK0 near WASHINGTON0 that accepts SOMETHING0 and has a SOMETHING1.",
    "Find NOTEBOOK0 near WASHINGTON0 that accepts SOMETHING0 and has a SCHOOL0.",
    "Find NOTEBOOK0 near WASHINGTON0 that accepts SCHOOL0 and has a WASHINGTON1.",
    "Find NOTEBOOK0 near WASHINGTON0 that accepts SCHOOL0 and has a SOMETHING0.",
    "Find NOTEBOOK0 near WASHINGTON0 that accepts SCHOOL0 and has a SCHOOL1."
]
four_script = [
    "[SEARCH ONE FROM:WASHINGTON0 WITH:WASHINGTON2 WITH:WASHINGTON3]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:WASHINGTON2 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:WASHINGTON2 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:SOMETHING0 WITH:WASHINGTON2]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:SOMETHING0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:SOMETHING0 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:SCHOOL0 WITH:WASHINGTON2]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:SCHOOL0 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:WASHINGTON0 WITH:SCHOOL0 WITH:SCHOOL1]",

    "[SEARCH ONE FROM:SOMETHING0 WITH:WASHINGTON1 WITH:WASHINGTON2]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:WASHINGTON1 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:WASHINGTON1 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:SOMETHING1 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:SOMETHING1 WITH:SOMETHING2]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:SOMETHING1 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:SCHOOL0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:SCHOOL0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:SOMETHING0 WITH:SCHOOL0 WITH:SCHOOL1]",

    "[SEARCH ONE FROM:SCHOOL0 WITH:WASHINGTON1 WITH:WASHINGTON2]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:WASHINGTON1 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:WASHINGTON1 WITH:SCHOOL1]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:SOMETHING0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:SOMETHING0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:SOMETHING0 WITH:SCHOOL1]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:SCHOOL1 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:SCHOOL1 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:SCHOOL0 WITH:SCHOOL1 WITH:SCHOOL2]",

    "[SEARCH ONE FROM:MONALISA0 WITH:WASHINGTON1 WITH:WASHINGTON2]",
    "[SEARCH ONE FROM:MONALISA0 WITH:WASHINGTON1 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:MONALISA0 WITH:WASHINGTON1 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:MONALISA0 WITH:SOMETHING0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:MONALISA0 WITH:SOMETHING0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:MONALISA0 WITH:SOMETHING0 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:MONALISA0 WITH:SCHOOL0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:MONALISA0 WITH:SCHOOL0 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:MONALISA0 WITH:SCHOOL0 WITH:SCHOOL1]",

    "[SEARCH ONE FROM:NOTEBOOK0 WITH:WASHINGTON1 WITH:WASHINGTON2]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:WASHINGTON1 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:WASHINGTON1 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:SOMETHING0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:SOMETHING0 WITH:SOMETHING1]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:SOMETHING0 WITH:SCHOOL0]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:SCHOOL0 WITH:WASHINGTON1]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:SCHOOL0 WITH:SOMETHING0]",
    "[SEARCH ONE FROM:NOTEBOOK0 WITH:SCHOOL0 WITH:SCHOOL1]"
]
four_class_id=[
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

script_total = [one_script, two_script, three_script, four_script]

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
entity_changed_sentence, replace_saved_dict, line_time, line_distance = find_and_change_entity(lines)

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
entity_num = len(replace_saved_dict)
if entity_num==0:
    entity_num=1
print('entity num : ', entity_num)
for i, message_embedding in enumerate(message_embeddings_total[entity_num-1]):
    error = rmse(np.array(message_embedding), np.array(test_message_embeddings))
    if minimum > error:
      minimum = error
      minimum_index = i

print("Minimum RMSE value: {}".format(minimum))
print("Most similar script: {}".format(script_total[entity_num-1][minimum_index]))
print("Estimation: {}".format(minimum_index))
#print("Answer: {}\n".format(test_label))
result2 = script_total[entity_num-1][minimum_index] #query

print("entity_number : ", entity_num)

result3 = replace_to_script(result2, replace_saved_dict, line_time, line_distance)
print("input: {}".format(lines))
print("Replace nouns: {}".format(replace_saved_dict))
print("Selected Sentence: {}".format(message_total[entity_num-1][minimum_index]))
print("Query: {}".format(result2))
print("classID : ", embeded_class_id[entity_num - 1][minimum_index])
print("Predict : ", result3)

time4 = time.time()

print('time0 = {}'.format(time1 - time0))
print('time1 = {}'.format(time2 - time1))
print('time2 = {}'.format(time3 - time2))
print('time3 = {}'.format(time4 - time3))
print('total time = {}'.format(time4 - time0))
time_end = time.time()






