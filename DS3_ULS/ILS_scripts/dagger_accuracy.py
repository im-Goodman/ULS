import re, os, glob, sys

files = glob.glob('*dagger*training_accuracy*')

acc_d = dict()
dagger_iter_list = []

for file in files :
    fp = open(file, 'r')
    matchobj = re.match(r'.*_dagger(.*)_training_accuracy.rpt', file)
    dagger_num = matchobj.group(1)

    for line in fp :
        line = line.strip()
        matchobj = re.match(r'(cluster(.*)) merged accuracy: (.*)', line)
        if matchobj :
            cluster = matchobj.group(1)
            cluster_num = matchobj.group(2)
            accuracy = matchobj.group(3)
            
            if not dagger_num in acc_d :
                acc_d[dagger_num] = dict()

            acc_d[dagger_num][cluster] = accuracy
            dagger_iter_list.append(int(dagger_num))

max_dagger_iter = max(dagger_iter_list)

cluster_list = ['clustera', 'cluster0', 'cluster1', 'cluster2', 'cluster3', 'cluster4']

for dagger_iter in range(max_dagger_iter) :
    dagger_iter = int(dagger_iter) + 1
    print(str(dagger_iter) + ' ', end='')
    for cluster in cluster_list :
        print(str(acc_d[str(dagger_iter)][cluster]) + ' ', end='')
    print()
