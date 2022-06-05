#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'minimumBribes' function below.
#
# The function accepts INTEGER_ARRAY q as parameter.
#

def minimumBribes(q):
    orig_q = q
    sorted_q = sorted(q)

    # print("orig: ", orig_q)
    # print("sorted: ", sorted_q)
    max_bribe = 2
    total_bribe_count = 0
    indices_checked = set()
    for idx, elem in enumerate(sorted_q):
        if elem in indices_checked:
            continue
        changed_idx = orig_q.index(elem)
        delta = 0
        # print("elem", elem)
        if changed_idx != idx:
            delta = abs(changed_idx - idx)
            # print("delta: ", delta)
            if delta > max_bribe:
                print("Too chaotic")
                return
            else:
                indices_checked.add(elem)
                indices_checked.add(orig_q[idx])
                total_bribe_count += 1
        # print("checked: ", indices_checked)
        # print("total_bribe_count ", total_bribe_count)
    print(total_bribe_count)


if __name__ == '__main__':
    t = int(input().strip())

    for t_itr in range(t):
        n = int(input().strip())

        q = list(map(int, input().rstrip().split()))

        minimumBribes(q)
