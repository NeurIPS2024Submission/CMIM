import os
import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--Res_dir', default='experiments/CIFAR10/kd_nasty_resnet18/CAT', type=str)
    args = parser.parse_args()
    for folder in os.listdir(args.Res_dir):
        FolderPath = os.listdir(os.path.join(args.Res_dir, folder))
        for file in FolderPath:
            if ".log" in file:
                with open(os.path.join(args.Res_dir, folder, file)) as f:
                    f = f.readlines()
                best_acc = -1
                for line in f:
                    if "best acc" in line:
                        ACC = float(line.split()[-1])
                        if ACC>best_acc:
                            best_acc = ACC
                print(folder, best_acc)
                
