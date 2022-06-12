import numpy as np
import config
import dnnlib
import dnnlib.tflib as tflib
import pickle

import PIL.Image
# load the StyleGAN model into Colab
URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)
# load the latents
s1 = np.load('emb/curdi_01.npy')
s2 = np.load('emb/elad_01.npy')
s3 = np.load('emb/iraqi_01.npy')
s4 = np.load('emb/polish_01.npy')
s5 = np.load('emb/ukrainian_01.npy')
# s6 = np.load('emb/59549772_10216054595290509_2957257503542345728_o_01.npy')
# s7 = np.load('emb/71500729_10157771709411340_621570437331025920_o_01.npy')
# s8 = np.load('emb/IMG-3813_01.npy')
s1 = np.expand_dims(s1,axis=0)
s2 = np.expand_dims(s2,axis=0)
s3 = np.expand_dims(s3,axis=0)
s4 = np.expand_dims(s4,axis=0)
s5 = np.expand_dims(s5,axis=0)
# s6 = np.expand_dims(s6,axis=0)
# s7 = np.expand_dims(s7,axis=0)
# s8 = np.expand_dims(+s8,axis=0)

# combine the latents somehow... let's try an average:
savg = ((s2*2)+s1+s3+s4+s5)/6
# run the generator network to render the latents:
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=5)
images = Gs_network.components.synthesis.run(savg, randomize_noise=False, **synthesis_kwargs)
(PIL.Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').resize((512,512),PIL.Image.LANCZOS)).save('interp.jpg','JPEG')
