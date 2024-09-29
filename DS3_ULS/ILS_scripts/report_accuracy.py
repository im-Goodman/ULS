import os, re, glob, sys, numpy as np

classifier = sys.argv[1]

file = open('./' + classifier + '_training_accuracy.rpt', 'r')

accuracy = dict()

for line in file :

    line = line.strip()
    matchobj = re.match(r'(.*) (.*) accuracy: (.*)', line)
    if matchobj :
        cluster = matchobj.group(1)
        scale = matchobj.group(2)
        acc = matchobj.group(3)
        if scale not in accuracy :
            accuracy[scale] = dict()
        
        accuracy[scale][cluster] = acc

for scale in accuracy :
    print(scale + '\t', end='')
    for cluster in ['clustera', 'cluster0', 'cluster1', 'cluster2', 'cluster3', 'cluster4'] :
        print(accuracy[scale][cluster] + '\t', end='')
    print()
