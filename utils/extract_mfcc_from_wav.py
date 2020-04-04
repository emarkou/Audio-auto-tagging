from python_speech_features import mfcc
import scipy.io.wavfile as wav


def wav_to_mfcc(wav_filename, num_cepstrum):
    """ extract MFCC features from a wav file

    :param wav_filename: filename with .wav format
    :param num_cepstrum: number of cepstrum to return
    :return: MFCC features for wav file
    """
    (rate, sig) = wav.read(wav_filename)
    mfcc_features = mfcc(sig, rate, numcep=num_cepstrum)
    return mrffc_features


