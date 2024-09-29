import sys
sys.path.append('./')

from DTPM_utils import *
import numpy as np
np.set_printoptions(suppress=True)


combos = multinomial_combinations(10,6)

for i in range(20) :
    np.random.shuffle(combos)
    print(combos[0:9], end='')
    print(', ', end='')
    #for combo in combos :
    #    print(combo)
    #    print(',', end='')
print()
