import re, os, glob, sys
import matplotlib

files = glob.glob('./reports/*dagger*')

max_dagger_iter = int(sys.argv[1])

exectime_d = dict()
dagger_iter_list = []

iteration_check_string = 'dagger' + str(max_dagger_iter + 1)
files = [file for file in files if iteration_check_string not in file]
# print(files)
# sys.exit(0)

for file in files :
    fp = open(file, 'r')
    matchobj = re.match(r'.*report(.*)_dagger(.*).rpt', file)
    scale      = matchobj.group(1)
    dagger_num = matchobj.group(2)

    for line in fp :
        line = line.strip()
        matchobj = re.match(r'.*Completed.*injection rate:(.*), conc.*ave_execution_time:(.*)', line)
        if matchobj :
            inj_rate = matchobj.group(1)
            exec_time = matchobj.group(2)
            
            if not dagger_num in exectime_d :
                exectime_d[dagger_num] = [[inj_rate, exec_time]]

            exectime_d[dagger_num].append([inj_rate, exec_time])
            dagger_iter_list.append(int(dagger_num))

# max_dagger_iter = max(dagger_iter_list)

length = len(exectime_d['1'])
for i in range(length) :
    for dagger_iter in range(max_dagger_iter) :
        dagger_iter = int(dagger_iter) + 1
        #print(str(dagger_iter) + ' ', end='')
        print(exectime_d[str(dagger_iter)][i][0] + ' ' + exectime_d[str(dagger_iter)][i][1] + ' ', end='')
    print()

