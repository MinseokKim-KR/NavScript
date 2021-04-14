from  more_itertools import unique_everseen

def replace_entity(sentences, scripts):
    place = ['burger king', 'burgerking', 'restroom', 'restroom facility',  'super market', 'supermarket', 'school', 'starbucks', 'coffee shop', 'cafe', 'college', 'gas station', 'park', 'department store', 'restaurent', 'indoor parking', 'outdoor parking', 'my destination']
    other = ['route', 'traffic camera', 'traffic', 'speed camera']
    weather = ['weather', 'weather forecast']
    geocode = ['US-101', 'Bayshore Blvd', 'I-580 East']
    event = ['meeting', 'party', 'speech']
    time = ['tomorrow', 'the day after tomorrow', 'next week', 'afternoon', 'morning', 'evening', 'dawn', 'midnight', 'noon']
    distance = ['50 meters', '100 meters', '500 meters', '1000 meters', '2000 meters']
    service = ['valet service', 'credit card']
    search_place = ['San Francisco Museum of Modern Art', 'Downtown Berkeley', 'Bay Bridge', 'Oakland', 'Seoul', 'New York', 'Wahinton', 'Chicago']
    Keywords=['[place]','[other]','[weather]','[geocode]','[event]','[time]','[distance]','[service]','[search_place]']
    entity =[place,other,weather,geocode,event,time,distance,service, search_place]
    class_lists = []
    g_sentence=[]
    g_scripts=[]
    number=0


    for sentence, script in zip(sentences, scripts):
        class_id = number
        replaced_sentence = [sentence.replace(key, entity[Keywords.index(key)][j],1)
                    for key in Keywords if (sentence.find(key) != -1)
                    for j in range(len(entity[Keywords.index(key)]))]
        replaced_script = [script.replace(key, entity[Keywords.index(key)][j],1)
                             for key in Keywords if (script.find(key) != -1)
                             for j in range(len(entity[Keywords.index(key)]))]

        for K in Keywords:
            print(sentence)
            if K in replaced_sentence[0]:
                temp = replaced_sentence
                temp2 = replaced_script
                replaced_sentence = [temp[a].replace(sentence, entity[Keywords.index(sentence)][j],1)
                                     for a in range(len(temp)) if (s in temp[a] for s in Keywords)
                                     for sentence in Keywords if (temp[a].find(sentence) != -1)
                                     for j in range(len(entity[Keywords.index(sentence)]))]
                replaced_script = [temp2[a].replace(sentence, entity[Keywords.index(sentence)][j],1)
                                   for a in range(len(temp2)) if (s in temp2[a] for s in Keywords)
                                   for sentence in Keywords if (temp2[a].find(sentence) != -1)
                                   for j in range(len(entity[Keywords.index(sentence)]))]
        replaced_sentence = list(unique_everseen(replaced_sentence))
        replaced_script = list(unique_everseen(replaced_script))
        classID = []
        for i in range(len(replaced_sentence)):
            classID.append(class_id)

        g_sentence =g_sentence+replaced_sentence
        g_scripts = g_scripts + replaced_script
        class_lists = class_lists + classID
        number = number +1

    return g_sentence, g_scripts, class_lists





def make_data():
    sentences = [
            "What's the [weather] for this [time]?",
            "What's the [other] like on my [other]?",
            "Show me a [other] on [geocode] and [geocode].",
            "Can you find me a [place] with [place] nearby?",
            "Find a [place] along route.",
            "Find the cheapest [place] within [distance] of [place].",
            "Okay, can you find me a [place] on my route that has a [place]?",
            "Find [place] near destination that accepts [service] and has a [service].",#Isn't it need range(near) query?

            "Navigate to [search_place].",
            "Reroute using [geocode].",

            "Drive to [search_place].",
            "Can I make [time] [event] without recharging?",
            "What's traffic like on the [search_place]?",
            "Are there any [other] on my route?",
            "Will it rain [time] in [search_place]?"
            ]

    navscripts = [
            "[SEARCH FROM:[weather] WHERE:HERE WHEN:[time]]",
            "[SEARCH FROM:[other] WHERE:[other]]",
            "[SEARCH FROM:[other] WHERE:[SEARCH GEOCODE WHERE:[geocode] and [geocode]]]",
            "[SEARCH FROM:[place] WHERE:NEARBY WITH:[place]]",
            "[SEARCH ONE FROM:[place] WHERE:ALONGROUTE]",
            "[SEARCH ONE FROM:[place] WHERE:[place] RANGE:[distance] WITH:[SORT PRICE ASC]]",
            "[SEARCH ONE FROM:[place] WHERE:ONROUTE WITH:[place]]",
            "[SEARCH ONE FROM:[place] WITH:[service] WITH:[service]]",

            "[ROUTE TO:[SEARCH KEYWORD:[search_place]]]",
            "[ROUTE ALTROUTE USE:[SEARCH LINKS:[geocode]]]",

            "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:[search_place]]]]",
            "[MODE DRIVERANGE TO:[SEARCH KEYWORD: [event] FROM:SCHEDULE WHEN:[time]] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
            "[MODE TRAFFIC [SEARCH FROM:TRAFFIC WHERE:[SEARCH KEYWORD:[search_place]] WITH:[VOICERESPONSE TEMPLATE:""*]",
            "[MODE [other] WHERE:ONROUTE WITH:[VOICERESPONSE TEMPLATE:""*]]",
            "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:[search_place]] WHEN:[time]]"
            ]
    # removed this sentences(special case). "What's my ETA to destination? ","[MODE DRIVERANGE]"
    sentence_result, script_result, class_result = replace_entity(sentences, navscripts)

    #### Add the special case in the end of the list
    special_sentences = ["What's my ETA to destination", "Show me alternative routes.","Show route overview.", "What's my drive range?"]
    special_scripts=["[ROUTE ALTROUTE]", "[ROUTE ALTROUTE]","[MODE ROUTEOVERVIEW]", "[MODE DRIVERANGE]"]
    number = class_result[-1]
    for sentence, script in zip(special_sentences, special_scripts):
        sentence_result = sentence_result + [sentence]
        script_result = script_result + [script]
        class_result = class_result + [number]
        number = number +1
    print(sentence_result)
    print(script_result)
    print(class_result)
    print(len(sentence_result))
    print(len(script_result))
    print(len(class_result))
    number = 0

    with open('Test_sentences.txt', 'w') as f:
        for sentence, navscript, classID in zip(sentence_result, script_result, class_result):
            # f.write("{}\n{}\n".format(sentence, navscript))
            f.write(str(number) + "||" + sentence + "||" + str(classID) + "||" + navscript + "\n")
            number = number +1

    return

if __name__ == "__main__":
    make_data()
