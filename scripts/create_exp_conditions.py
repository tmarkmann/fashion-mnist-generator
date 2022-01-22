import csv
import glob

ROOTDIR = './exp_images'
HEADER = ['type', 'path']

with open(f'{ROOTDIR}/conditions.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)

    real_images = glob.glob(f'{ROOTDIR}/real/*.png')
    for file in real_images[:10]:
        writer.writerow(['real', file])

    #for file in glob.glob(f'{ROOTDIR}/tfgan/*.png'):
    #    writer.writerow(['tfgan', file])
    #
    #for file in glob.glob(f'{ROOTDIR}/vae/*.png'):
    #    writer.writerow(['vae', file])
    #
    #for file in glob.glob(f'{ROOTDIR}/lsgm/*.png'):
    #    writer.writerow(['lsgm', file])
    #for file in glob.glob(f'{ROOTDIR}/stylegan2/*.png'):
    #    writer.writerow(['stylegan2', file])