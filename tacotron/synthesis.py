#-*- coding: utf-8 -*-

from network import *
from data import inv_spectrogram, find_endpoint, save_wav, spectrogram
import numpy as np
import argparse
import os, sys
import io
from text import text_to_sequence

use_cuda = torch.cuda.is_available()

try: # older versions of torch don't recognise mps
    use_mps = torch.backends.mps.is_available() & torch.backends.mps.is_built()
except:
    use_mps = False

def main(args):

    # Make model
    if use_cuda:
        model = nn.DataParallel(Tacotron().cuda())
    else:
        model = Tacotron()

    # Load checkpoint
    try:
        if use_cuda:
            checkpoint = torch.load(os.path.join(hp.checkpoint_path,'checkpoint_%d.pth.tar'% args.restore_step))
        else:
            checkpoint = torch.load(os.path.join(hp.checkpoint_path,'checkpoint_%d.pth.tar'% args.restore_step), map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint['model'])
        print("\n--------model restored at step %d--------\n" % args.restore_step)

    except:
        raise FileNotFoundError("\n------------Model not exists------------\n")

    # Evaluation
    model = model.eval()

    # Make result folder if not exists
    if not os.path.exists(hp.output_path):
        os.mkdir(hp.output_path)

    # Sentences for generation
    sentences = [
        "And it is worth mention in passing that, as an example of fine typography,",
        # From July 8, 2017 New York Times:
        'Scientists at the CERN laboratory say they have discovered a new particle.',
        'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
        'President Trump met with other leaders at the Group of 20 conference.',
        'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
        # From Google's Tacotron example page:
        'Generative adversarial network or variational auto-encoder.',
        'The buses aren\'t the problem, they actually provide a solution.',
        'Does the quick brown fox jump over the lazy dog?',
        'Talib Kweli confirmed to All HipHop that he will be releasing an album in the next year.',
    ]

    # Synthesis and save to wav files
    for i, text in enumerate(sentences):
        wav = generate(model, text)
        path = os.path.join(hp.output_path, 'result_%d_%d.wav' % (args.restore_step, i+1))
        with open(path, 'wb') as f:
            f.write(wav)

        f.close()
        print("save wav file at step %d ..." % (i+1))

def generate(model, text):

    with torch.no_grad():
        # Text to index sequence
        cleaner_names = [x.strip() for x in hp.cleaners.split(',')]
        seq = np.expand_dims(np.asarray(text_to_sequence(text, cleaner_names), dtype=np.int32), axis=0)

        # Provide [GO] Frame
        mel_input = np.zeros([seq.shape[0], hp.num_mels, 1], dtype=np.float32)

        # Variables
        characters = Variable(torch.from_numpy(seq).type(torch.cuda.LongTensor)).cuda()
        mel_input = Variable(torch.from_numpy(mel_input).type(torch.cuda.FloatTensor)).cuda()

        # Spectrogram to wav
        _, linear_output = model.forward(characters, mel_input)
        wav = inv_spectrogram(linear_output[0].data.cpu().numpy())
        wav = wav[:find_endpoint(wav)]
        out = io.BytesIO()
        save_wav(wav, out)

        return out.getvalue()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, help='Global step to restore checkpoint', default=4000)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=1)
    args = parser.parse_args()
    main(args)
