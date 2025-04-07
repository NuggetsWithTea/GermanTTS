import edge_tts
import asyncio

from time import sleep
from os import getcwd, makedirs, system, environ
from os.path import isdir

#used directories
base_dir = getcwd() 
audio_dir = base_dir + "\\generated_audio"

if not isdir(audio_dir):
    makedirs(audio_dir)


async def generate(input):
    tts = edge_tts.Communicate(
        text=input,
        voice="de-DE-KatjaNeural",
    )
    #save audio as wav file
    output_file_name = input[0:10].strip().replace(" ", "_")
    result_path = audio_dir + f"\\{output_file_name}.wav"
    await tts.save(result_path)
    print(f"saved as: {result_path}")
    sleep(3)

while 1:
    system("cls")
    text = input("enter prompt to generate auido for:")
    asyncio.run(generate(text))
