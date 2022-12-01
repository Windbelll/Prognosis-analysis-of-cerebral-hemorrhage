import os
import numpy as np
# settings total = 100
train_rate = 80
val_rate = 10
test_rate = 10

if not os.path.exists("./bad"):
    print("error: no bad data")
if not os.path.exists("./good"):
    print("error: no good data")
train = open("./train.txt", 'w')
val = open("./val.txt", 'w')
test = open("./test.txt", 'w')

bad_names = os.listdir("./bad")
for name in bad_names:
    choice = np.random.randint(100)
    if choice < train_rate:
        train.write("data/bad/" + name + "  " + str(0) + "\n")
    elif choice < train_rate + val_rate:
        val.write("data/bad/" + name + "  " + str(0) + "\n")
    else:
        test.write("data/bad/" + name + "  " + str(0) + "\n")

good_names = os.listdir("./good")
for name in good_names:
    choice = np.random.randint(100)
    if choice < train_rate:
        train.write("data/good/" + name + "  " + str(1) + "\n")
    elif choice < train_rate + val_rate:
        val.write("data/good/" + name + "  " + str(1) + "\n")
    else:
        test.write("data/good/" + name + "  " + str(1) + "\n")



