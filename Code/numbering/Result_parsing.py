#f1=open("Result1_500.txt",'r')
#f2=open("Result2_500.txt",'r')
#f3=open("Result3_500.txt",'r')
#f4=open("Result4_500.txt",'r')
#f5=open("Result5_500.txt",'r')
#lines=[f1.readlines(), f2.readlines(), f3.readlines(), f4.readlines(), f5.readlines()]
f1=open("Result1_500.txt",'r')
f2=open("Result2_500.txt",'r')
f3=open("Result3_500.txt",'r')
f4=open("Result4_500.txt",'r')
lines=[f1.readlines(), f2.readlines(), f3.readlines(), f4.readlines()]
#print(len(lines[0]))
#print(len(lines[1]))
#print(len(lines[2]))
#print(len(lines[3]))
#print(len(lines[4]))
w1=open("./dataset/class_wrong_script_wrong.txt",'w')
w2=open("./dataset/class_wrong_script_correct.txt",'w')
w3=open("./dataset/class_correct_script_wrong.txt",'w')
w4=open("./dataset/class_correct_script_correct.txt",'w')
count = 0
for i in range(len(lines)):
    for L in lines[i]:
        sp = L.replace('\n','').split("||")
        if sp[5] == 'X' and sp[6] == 'X':
            w1.write(L)
        elif sp[5] == 'X' and sp[6] == 'O':
            w2.write(L)
        elif sp[5] == 'O' and sp[6] == 'X':
            w3.write(L)
            #print(count)
            #count = count + 1
        elif sp[5] == 'O' and sp[6] == 'O':
            w4.write(L)
f1.close()
f2.close()
f3.close()
f4.close()
w1.close()
w2.close()
w3.close()
w4.close()
