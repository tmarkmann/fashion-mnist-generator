from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import glob
from utils.extract_npz_images import write_npz_to_images

flags.DEFINE_string('npz_dir', '/tmp/npz', 'Directory where to find npz files.')

flags.DEFINE_string('output_dir', '/tmp/lsgm_out/',
                    'Directory to extract npz files to.')

FLAGS = flags.FLAGS


def main(_):
    npz_files = glob.glob(f'{FLAGS.npz_dir}/*.npz')
    write_npz_to_images(npz_paths=npz_files, output_dir=FLAGS.output_dir)  

if __name__ == '__main__':
    app.run(main)
