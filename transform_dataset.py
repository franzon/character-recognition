import os
from shutil import copyfile

for i in ['Test', 'Train', 'Valid']:
    file = open('./character_dataset/NIST_{}_Upper.txt'.format(i))
    lines = file.readlines()
    file.close()

    for line in lines:
        line = line.strip()
        c = line.split('/')[1].upper()
        f = line.split('/')[2]

        if not os.path.exists('./dataset/{}/'.format(i)+c):
            os.makedirs('./dataset/{}/'.format(i)+c)

        copyfile('./character_dataset/' + c + '/' + f,
                 './dataset/{}/'.format(i) + c + '/' + f)
