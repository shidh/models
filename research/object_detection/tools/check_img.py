
import os
import sys
from PIL import Image

train_dir = sys.argv[1]

def progress(percent, width=50):
    '''print progress'''
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  #concate string
    print('\r%s %d%% ' % (show_str, percent), end='')

def is_valid_jpg(jpg_file):
    with open(jpg_file, 'rb') as f:
        f.seek(-2, 2)
        buf = f.read()
        f.close()
        return buf ==  b'\xff\xd9'  #check img file contains end file

def is_jpg(filename):
    try:
        i=Image.open(filename)
        return i.format =='JPEG'
    except IOError:
        print(filename)
        print('\rfile path: %s ' %  (train_dir + file))
        return False

from skimage import io
def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        print('\rincomplete file path: %s ' %  (train_dir + file))
        return False
    return True

data_size = len([lists for lists in os.listdir(train_dir) if (os.path.isfile(os.path.join(train_dir, lists)) and os.path.splitext(os.path.join(train_dir, lists))[1].lower() == '.jpg') ])
recv_size = 0
incompleteFile = 0
print('file tall : %d' % data_size)

for file in os.listdir(train_dir):
    if os.path.splitext(file)[1].lower() == '.jpg':
        ret = is_valid_jpg(train_dir + file)
        if ret == False:
            print('\rincomplete file path: %s ' %  (train_dir + file))
            incompleteFile = incompleteFile + 1
            #os.remove(train_dir + file)

        ret = is_jpg(train_dir + file)
        if ret == False:
            print('\rincomplete file path: %s ' %  (train_dir + file))

        ret = verify_image(train_dir + file)
        if ret == False:
            print('\rincomplete file path: %s ' %  (train_dir + file))

    recv_per = int(100 * recv_size / data_size)
    progress(recv_per, width=30)
    recv_size = recv_size + 1

progress(100, width=30)
print('\nincomplete file : %d' % incompleteFile)
