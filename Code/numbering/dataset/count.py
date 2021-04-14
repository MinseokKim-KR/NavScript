import sys
classID=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
f1=open(sys.argv[1],'r')
lines=f1.readlines()
for line in lines:
    sp = line.split("||")
    classID[int(sp[1])] = classID[int(sp[1])] + 1
print(classID)
