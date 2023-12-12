import os
import random
import shutil

def _transfer_content(files, src_dir, backup_dir, mode):
    for f in files:
        src = os.path.join(src_dir, f)
        dst = os.path.join(backup_dir, f)

        if mode == "move":
            shutil.move(src, dst)
        elif mode == "copy":
            shutil.copy(src, dst)


def create_backup_directories(subdir="speed_limits"):
    """ Creates a backup directory of the given class subdirectory """
    test_dir = "test/%s" %subdir
    train_dir = "train/%s" %subdir

    test_backup_dir = "%s_test_backup" %subdir
    train_backup_dir = "%s_train_backup" %subdir

    if os.path.exists(test_backup_dir) or os.path.exists(train_backup_dir):
        print("Directory '%s' already exists. Either backup and delete the contents or choose another name.")
        exit(0)

    os.makedirs(test_backup_dir)
    os.makedirs(train_backup_dir)

    test_files = os.listdir()
    train_files = os.listdir()

    _transfer_content(test_files, test_dir, test_backup_dir, mode="copy")
    _transfer_content(train_files, train_dir, train_backup_dir, mode="move")


def split_and_inject_samples(subdir="speed_limits", inject_size=0.7):
    """ Split the contents of a source directory and transfer them to another directory """
    test_dir = "test/%s" %subdir
    train_dir = "train/%s" %subdir

    src_files = os.listdir(test_dir)

    files_sample = random.sample(src_files, int(inject_size * len(src_files)))

    _transfer_content(files_sample, test_dir, train_dir, mode="move")


def main():
    create_backup_directories()
    split_and_inject_samples()


if __name__ == "__main__":
    main()
