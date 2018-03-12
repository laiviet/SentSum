import sys
import utils_eval
import os

if len(sys.argv) > 1:
    output_path = sys.argv[1]
    folders = []
    for folder in os.listdir(output_path):
        try:
            epoch = int(folder)
            if os.path.isdir(output_path + folder):
                epoch = int(folder)
                folders.append((epoch, folder))
        except:
            print("Ignore folder: " + folder)
    folders.sort()
    list_thread = []
    for epoch, folder in folders:
        print(output_path + folder)
        utils_eval.evaluate(output_path + folder)

else:
    print('Please use following command\n# python evalall.py [directory]')
