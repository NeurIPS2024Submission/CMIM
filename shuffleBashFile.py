import os
import random
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--BashFile', default='script/tinyimagenet/TinScratch_20.sh', type=str)
    args = parser.parse_args()
    f = open(args.BashFile, "r")
    Lines = f.readlines()
    random.shuffle(Lines)
    f = open(args.BashFile, "w")
    for L in Lines:
        f.write(L)
