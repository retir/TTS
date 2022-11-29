python downloader.py
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1

mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt
mv train.txt data/
tar -xvf mel.tar.gz
echo $(ls mels | wc -l)

unzip alignments.zip >> /dev/null
unzip energies.zip >> /dev/null
unzip LJSpeech.zip >> /dev/null
unzip pitches.zip >> /dev/null