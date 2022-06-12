
# embed and check
# embed non-face and check
# create n images by intepolating m verctors with uniform wwights but non higher than x

import math
import pickle
import PIL.Image
import config
import dnnlib
import dnnlib.tflib as tflib
from encoder.generator_model import Generator
import numpy as np
import os
import cv2
from datetime import datetime
from time import time


max_people = 15 #use None for no limitation
steps = 50 #normally 200-300
spherical = True

interpolated_images_dir = 'emb'
output_dir = 'gen_videos'
dlatent_dir = 'emb'
latent1_file_name = 'elad_01.npy'


URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

use_sin = True
def shift_sin(x):
    if not use_sin:
        return x
    return (np.sin((x-0.5)*np.pi)+1)/2

def interp(steps, low, high, alpha=0, spherical=False):
    vals = [shift_sin((step*(1-alpha)) / (steps + 1)) for step in range(steps + 2)]

    ret = np.asarray([(1.0 - val) * low + val * high for val in vals])  # L'Hopital's rule/LERP
    if not spherical:
        return ret
    omega = np.arccos(np.clip(
        np.sum(low / np.linalg.norm(low, axis=-1, keepdims=True) * high / np.linalg.norm(high, axis=-1, keepdims=True),
               axis=-1, keepdims=True), -1, 1))
    so = np.sin(omega)
    ind = so[:, 0] != 0
    ret[:, ind] = np.asarray(
        [np.sin((1.0 - val) * omega[ind]) / so[ind] * low[ind] + np.sin(val * omega[ind]) / so[ind] * high[ind] for val
         in vals])
    return ret


generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

model_res = 1024
model_scale = int(2 * (math.log(model_res, 2) - 1))


def generate_raw_image(latent_vector):
    latent_vector = latent_vector.reshape((1, model_scale, 512))
    generator.set_dlatents(latent_vector)
    return generator.generate_images()[0]


def generate_image(latent_vector):
    img_array = generate_raw_image(latent_vector)
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img


video_out = cv2.VideoWriter(os.path.join(output_dir, datetime.now().strftime("%d_%m_%Y__%H_%M_%S")+".avi"), cv2.VideoWriter_fourcc(*'MJPG'), 30, (model_res, model_res))

elad_vec = np.load(os.path.join(dlatent_dir, latent1_file_name))
latent1 = elad_vec

beta = 0.7
dynamic_beta = True
dynamic_beta_weight = 100

img_files = []
for img_name in os.listdir(dlatent_dir)[:max_people]:
    if img_name.endswith('.npy') and img_name != latent1_file_name:
        img_files.append(os.path.join(dlatent_dir, img_name))
print('found %d vector files'%len(img_files))
regress_to_elad = True
regression_fraction = 2/3
frame_to_start_regression = int(len(img_files)*regression_fraction)
regression_steps = len(img_files)-frame_to_start_regression

for j,img_name in enumerate(img_files):
    start = time()
    latent2 = np.load(img_name)

    print("interpolating image (%d): %s"%(j+1,img_name))
    vectors = interp(steps, latent1, latent2, spherical=spherical)
    stopframe = int((steps+1)*beta)


    mirror = vectors[-2:0:-1][:stopframe]
    if regress_to_elad and j>=frame_to_start_regression: # compute new mirror that mixes with a little elad on the way back
        mirror = interp(len(mirror)-2, mirror[0], interp(regression_steps-2, mirror[-1], elad_vec, spherical=spherical)[j-frame_to_start_regression], spherical=spherical)

    vectors = np.vstack([vectors, mirror])

    for i, vector in enumerate(vectors):
        #print('%d/%d' % (i + 1, len(vectors)))
        np.expand_dims(vector, axis=0)
        # img = generate_image(vector)
        # img.save(os.path.join(interpolated_images_dir,str(i)+'.png'), 'PNG')
        #video_out.write(cv2.cvtColor(np.array(img).astype('uint8'), cv2.COLOR_RGB2BGR))
        img = generate_raw_image(vector)
        video_out.write(img[...,::-1])
    latent1 = vectors[-1]
    if dynamic_beta:
        beta = (beta*dynamic_beta_weight+1)/(dynamic_beta_weight+1)
    print('took %.2f minutes (%d frames)'%((time()-start)/60, len(vectors)))

video_out.release()

