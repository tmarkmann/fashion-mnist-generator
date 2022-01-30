import csv
import glob
from random import shuffle


ROOTDIR = '/Users/tmarkmann/fashion/fashion-mnist-exp/exp_images'
HEADER = ['type', 'path']

with open(f'{ROOTDIR}/conditions.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)

    real_images = glob.glob(f'{ROOTDIR}/real/*.png')
    shuffle(real_images)
    for file in real_images[:800]:
        file = '/'.join((file.split('/'))[-3:])
        writer.writerow(['real', file])

    tfgan_images = glob.glob(f'{ROOTDIR}/tfgan/*.png')
    shuffle(tfgan_images)
    for file in tfgan_images[:300]:
        file = '/'.join((file.split('/'))[-3:])
        writer.writerow(['tfgan', file])
    
    cvae_images = glob.glob(f'{ROOTDIR}/cvae/*.png')
    shuffle(cvae_images)
    for file in cvae_images[:300]:
        file = '/'.join((file.split('/'))[-3:])
        writer.writerow(['cvae', file])
    
    lsgm_images = glob.glob(f'{ROOTDIR}/lsgm/*.png')
    shuffle(lsgm_images)
    for file in lsgm_images[:300]:
        file = '/'.join((file.split('/'))[-3:])
        writer.writerow(['lsgm', file])
    
    stylegan_images = glob.glob(f'{ROOTDIR}/stylegan/*.png')
    shuffle(stylegan_images)
    for file in stylegan_images[:300]:
        file = '/'.join((file.split('/'))[-3:])
        writer.writerow(['stylegan2', file])