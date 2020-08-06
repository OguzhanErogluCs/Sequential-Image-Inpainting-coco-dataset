from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from PIL import Image
import matplotlib.image
import cv2
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = '/home/oguzhan1998ax/Downloads/annotations_trainval2017'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' ,'.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' ,'.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=["cat", "laptop"]);
imgIds = coco.getImgIds(catIds=catIds );
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

I = io.imread(img['coco_url'])
matplotlib.image.imsave("input.jpg", I)

#kernel = np.ones((3, 3), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(anns_ids)
mask = coco.annToMask(anns[0])
dilation = cv2.dilate(mask, kernel, iterations=2)
mask = 255. - mask
dilation = 255. - dilation
#dilation = cv2.dilate(dilation, kernel, iterations=10)
matplotlib.image.imsave('output.jpg', mask, cmap="gray")
matplotlib.image.imsave('outputt.jpg', dilation, cmap="gray")
for i in range(len(anns)):
    mask = coco.annToMask(anns[i])
    dilation = cv2.dilate(mask, kernel, iterations=2)
    mask = 255. - mask
    dilation = 255. - dilation
    #dilation = cv2.dilate(dilation, kernel, iterations=10)
    occurrences = np.count_nonzero(mask == 255)
    occurrences = (((mask.shape[0] * mask.shape[1]) - occurrences)/(mask.shape[0] * mask.shape[1])) * 100

    print("mask: ", occurrences)
    matplotlib.image.imsave("output"+str(i)+".jpg", mask, cmap="gray")
    matplotlib.image.imsave("outputt" + str(i) + ".jpg", dilation, cmap="gray")
"""import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
II = io.imread("/home/oguzhan1998ax/PycharmProjects/405/output.jpg")
I = io.imread("/home/oguzhan1998ax/PycharmProjects/sample-imageinpainting-HiFill/samples/maskset/1.jpg")

occurrences = np.count_nonzero(II == 255)
occurrences2 = np.count_nonzero(II == 0)
print(occurrences2+occurrences)

print(II.shape)"""
