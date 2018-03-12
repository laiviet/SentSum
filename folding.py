import os, shutil



def create_folder(path):
    try:
        os.mkdir(path)
    except:
        print('Folder existed: ', path)

def folding(source_path, dest_path, fold_ratio = '8:1:1'):
    files = os.listdir(source_path)
    enable_valid  = True
    denominator = 10
    train = []
    valid = []
    test  = []
    for idx, fname in enumerate(files):
        if idx % denominator == 0:
            test.append(fname)
        elif idx % denominator ==1:
            if enable_valid:
                valid.append(fname)
            else:
                train.append(fname)
        else:
            train.append(fname)

    train_path = os.path.join(dest_path, 'train')
    create_folder(train_path)
    for fname in train:
        src_path = os.path.join(source_path, fname)
        dst_path = os.path.join(train_path, fname)
        shutil.copy(src_path, dst_path)

    test_path = os.path.join(dest_path, 'test')
    create_folder(test_path)
    for fname in test:
        src_path = os.path.join(source_path, fname)
        dst_path = os.path.join(test_path, fname)
        shutil.copy(src_path, dst_path)

    if enable_valid:
        valid_path = os.path.join(dest_path, 'valid')
        create_folder(valid_path)
        for fname in valid:
            src_path = os.path.join(source_path, fname)
            dst_path = os.path.join(valid_path, fname)
            shutil.copy(src_path, dst_path)



folding('data/parsedata', 'data')