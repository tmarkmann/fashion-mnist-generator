import csv
import glob

ROOTDIR = '/Users/tmarkmann/fashion/fashion-mnist-exp/exp_images'
HEADER = ['type', 'path']

with open(f'{ROOTDIR}/conditions.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)

    real_images = glob.glob(f'{ROOTDIR}/real/*.png')
    for file in real_images[:10]:
        writer.writerow(['real', file])

    tfgan_images = glob.glob(f'{ROOTDIR}/tfgan/*.png')
    for file in tfgan_images[:10]:
        writer.writerow(['tfgan', file])
    
    cvae_images = glob.glob(f'{ROOTDIR}/cvae/*.png')
    for file in cvae_images[:10]:
        writer.writerow(['cvae', file])
    
    lsgm_images = glob.glob(f'{ROOTDIR}/lsgm/*.png')
    for file in lsgm_images[:10]:
        writer.writerow(['lsgm', file])
    
    #for file in glob.glob(f'{ROOTDIR}/stylegan2/*.png'):
    #    writer.writerow(['stylegan2', file])