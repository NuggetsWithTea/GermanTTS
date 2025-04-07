import torchaudio
from torch import tensor
from time import sleep
from os import getcwd, makedirs, system, environ
from os.path import isdir

from torchaudio.functional import gain
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference import TTS
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.inference.enhancement import SpectralMaskEnhancement

#Hallo, ich bin Robert. Ich erkläre die heute, wie VEMAGS funktioniert.
# dataset: https://zenodo.org/records/7265581

#models
GERMAN_MODEL = "tensorspeech/tts-tacotron2-thorsten-ger"#padmalcom/tts-tacotron2-german"
ENGLISH_MODEL = "speechbrain/tts-tacotron2-ljspeech"

#disable strange warning 
environ["HF_HUB_DOWNLOAD_SYMLINKS"] = "0"
environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

#sample rate based on models training data (default: 22050)
SAMPLE_RATE = 22050
REFINEMENT_SAMPLE_RATE = 16000
OUTPUT_FORMAT = "wav"

WINDOWS_UNSUPPORTED_FILENAME_CHARS = ["<", ">", ":", ",", '"', "/", "\\", "|", "?", "*"]

#used directories
base_dir = getcwd() 
audio_dir_unrefined = base_dir + "\\generated_audio_unrefined"
audio_dir = base_dir + "\\generated_audio"

if not isdir(audio_dir_unrefined):
    makedirs(audio_dir_unrefined)

if not isdir(audio_dir):
    makedirs(audio_dir)

torchaudio.set_audio_backend("soundfile")

print("----------loading models----------")
#load text to spectrogram model
taco_ger = TTS.Tacotron2.from_hparams(
    source= GERMAN_MODEL,
    savedir=None,
    local_strategy=LocalStrategy.COPY,
)

taco_eng = TTS.Tacotron2.from_hparams(
    source= ENGLISH_MODEL,
    savedir=None,
    local_strategy=LocalStrategy.COPY,
)

#load vocoder model (mel-spectrogram to "real" audio)
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir=None,
    local_strategy=LocalStrategy.COPY
)

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir=None
)

print("---------------done---------------")
while 1:
    #clear terminal
    system("cls")

    #take input
    text_to_convert = input("enter text to generate speech with: ['/eng' in prompt for english text]\n")
    make_english = "/eng" not in text_to_convert
    model = taco_ger if make_english else taco_eng

    if make_english:
        text_to_convert.replace("/eng", "")

    #input to spectrogram
    mel_output, mel_length, alignment = model.encode_text(text_to_convert)

    #spectrogram to audio 
    decoded_batch = hifi_gan.decode_batch(mel_output)

    #clean up filename
    text_len = len(text_to_convert)
    output_file_name = ""
    if text_len > 25:
        output_file_name = text_to_convert[0:25].strip().replace(" ", "_")
    else:
        output_file_name = text_to_convert.strip().replace(" ", "_")

    for char in WINDOWS_UNSUPPORTED_FILENAME_CHARS:
        output_file_name = output_file_name.replace(char, "")

    #save audio as wav file
    unrefined_audio_path = audio_dir_unrefined + f"\\{output_file_name}.{OUTPUT_FORMAT}"
    torchaudio.save(
        unrefined_audio_path,
        decoded_batch.squeeze(1), 
        SAMPLE_RATE,
        format = OUTPUT_FORMAT
        )

    print(f"audio (unrefined) saved as: {unrefined_audio_path}")

    noisy = enhance_model.load_audio(f"\\generated_audio_unrefined\\{output_file_name}.{OUTPUT_FORMAT}")

    #resample audio to REFINEMENT_SAMPLE_RATE
    resampler = torchaudio.transforms.Resample(SAMPLE_RATE, REFINEMENT_SAMPLE_RATE)
    converted_noisy = resampler(noisy).unsqueeze(0)
    # Add a relative length tensor
    enhanced = enhance_model.enhance_batch(converted_noisy, lengths=tensor([1.0]))

    result_path = audio_dir + f"\\{output_file_name}.{OUTPUT_FORMAT}"
    
    #normalize refined audio
    normalized = gain(enhanced.cpu(), -3.0)

    # Save the enhanced audio to a file
    torchaudio.save(
        result_path, 
        normalized, 
        REFINEMENT_SAMPLE_RATE
    )
    print(f"audio (final) saved as: {result_path}")
    sleep(3)

