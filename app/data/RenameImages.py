import os

# change the path to wherever the images are that you would like to rename
path = './images/train/waterfalls'

i = 0
for filename in os.listdir(path):
    # Then change the name of the each image file . . . . . . . here.
    os.rename(os.path.join(path, filename), os.path.join(path, 'waterfall'+str(i) + '.jpg'))
    i = i + 1
