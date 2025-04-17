import numpy as np
import os
import librosa
import librosa_test
import sys
import csv

LOADED_SONGS = 20

def add_to_stft_csv(bass_stft, drums_stft, chords_stft, vocals_stft):
    centroid_path = os.path.join(os.getcwd(), "shivers", "stft")
    bc_path = os.path.join(centroid_path,"bc.csv")
    bd_path = os.path.join(centroid_path, "bd.csv")
    bv_path = os.path.join(centroid_path,"bv.csv")
    cd_path = os.path.join(centroid_path,"cd.csv")
    cv_path = os.path.join(centroid_path,"cv.csv")
    dv_path = os.path.join(centroid_path,"dv.csv")

    bc = np.mean(librosa_test.corr_coef_split(bass_stft,chords_stft))
    bd = np.mean(librosa_test.corr_coef_split(bass_stft,drums_stft))
    bv = np.mean(librosa_test.corr_coef_split(bass_stft,vocals_stft))
    cd = np.mean(librosa_test.corr_coef_split(chords_stft,drums_stft))
    cv = np.mean(librosa_test.corr_coef_split(chords_stft,vocals_stft))
    dv = np.mean(librosa_test.corr_coef_split(drums_stft,vocals_stft))

    with open(bc_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bc])

    with open(bd_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bd])

    with open(bv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bv])

    with open(cd_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([cd])

    with open(cv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([cv])
    
    with open(dv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([dv])



def add_to_centroid_csv(bass_centroid, drums_centroid, chords_centroid, vocals_centroid):
    centroid_path = os.path.join(os.getcwd(), "shivers", "centroid")
    bc_path = os.path.join(centroid_path,"bc.csv")
    bd_path = os.path.join(centroid_path, "bd.csv")
    bv_path = os.path.join(centroid_path,"bv.csv")
    cd_path = os.path.join(centroid_path,"cd.csv")
    cv_path = os.path.join(centroid_path,"cv.csv")
    dv_path = os.path.join(centroid_path,"dv.csv")

    bc = np.mean(librosa_test.corr_coef_split(bass_centroid,chords_centroid))
    bd = np.mean(librosa_test.corr_coef_split(bass_centroid,drums_centroid))
    bv = np.mean(librosa_test.corr_coef_split(bass_centroid,vocals_centroid))
    cd = np.mean(librosa_test.corr_coef_split(chords_centroid,drums_centroid))
    cv = np.mean(librosa_test.corr_coef_split(chords_centroid,vocals_centroid))
    dv = np.mean(librosa_test.corr_coef_split(drums_centroid,vocals_centroid))

    with open(bc_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bc])

    with open(bd_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bd])

    with open(bv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bv])

    with open(cd_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([cd])

    with open(cv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([cv])
    
    with open(dv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([dv])



def add_to_chroma_csv(bass_chroma, chords_chroma, vocals_chroma):
    chroma_path = os.path.join(os.getcwd(), "shivers", "chroma")
    bc_path = os.path.join(chroma_path,"bc.csv")
    bv_path = os.path.join(chroma_path,"bv.csv")
    cv_path = os.path.join(chroma_path,"cv.csv")

    bc = np.mean(librosa_test.corr_coef_split(bass_chroma,chords_chroma))
    bv = np.mean(librosa_test.corr_coef_split(bass_chroma,vocals_chroma))
    cv = np.mean(librosa_test.corr_coef_split(chords_chroma,vocals_chroma))

    with open(bc_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bc])

    with open(bv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bv])

    with open(cv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([cv])


def add_to_tonnetz_csv(bass_tonnetz, chords_tonnetz, vocals_tonnetz):
    chroma_path = os.path.join(os.getcwd(), "shivers", "tonnetz")
    bc_path = os.path.join(chroma_path,"bc.csv")
    bv_path = os.path.join(chroma_path,"bv.csv")
    cv_path = os.path.join(chroma_path,"cv.csv")

    bc = np.mean(librosa_test.corr_coef_split(bass_tonnetz,chords_tonnetz))
    bv = np.mean(librosa_test.corr_coef_split(bass_tonnetz,vocals_tonnetz))
    cv = np.mean(librosa_test.corr_coef_split(chords_tonnetz,vocals_tonnetz))

    with open(bc_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bc])

    with open(bv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bv])

    with open(cv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([cv])



def add_to_rms_csv(bass_rms, drums_rms, chords_rms, vocals_rms):
    centroid_path = os.path.join(os.getcwd(), "shivers", "rms")
    bc_path = os.path.join(centroid_path,"bc.csv")
    bd_path = os.path.join(centroid_path, "bd.csv")
    bv_path = os.path.join(centroid_path,"bv.csv")
    cd_path = os.path.join(centroid_path,"cd.csv")
    cv_path = os.path.join(centroid_path,"cv.csv")
    dv_path = os.path.join(centroid_path,"dv.csv")

    bc = np.mean(librosa_test.corr_coef_split(bass_rms,chords_rms))
    bd = np.mean(librosa_test.corr_coef_split(bass_rms,drums_rms))
    bv = np.mean(librosa_test.corr_coef_split(bass_rms,vocals_rms))
    cd = np.mean(librosa_test.corr_coef_split(chords_rms,drums_rms))
    cv = np.mean(librosa_test.corr_coef_split(chords_rms,vocals_rms))
    dv = np.mean(librosa_test.corr_coef_split(drums_rms,vocals_rms))

    with open(bc_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bc])

    with open(bd_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bd])

    with open(bv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([bv])

    with open(cd_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([cd])

    with open(cv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([cv])
    
    with open(dv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([dv])



def find_avg(folder_name):
    for folder in os.listdir():
        if folder == folder_name:
            if folder == "chroma" or folder == "tonnetz":
                bc_path = os.path.join(folder_name,"bc.csv")
                bv_path = os.path.join(folder_name,"bv.csv")
                cv_path = os.path.join(folder_name,"cv.csv")
                
                
                with open(bc_path) as file:
                    sum = 0
                    for row in csv.reader(file):
                        sum += float(row[0])
                    sum = sum/LOADED_SONGS

                with open(bc_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    file.truncate()
                    writer.writerow([sum])

                with open(bv_path) as file:
                    sum = 0
                    for row in csv.reader(file):
                        sum += float(row[0])
                    sum = sum/LOADED_SONGS

                with open(bv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    file.truncate()
                    writer.writerow([sum])

                with open(cv_path) as file:
                    sum = 0
                    for row in csv.reader(file):
                        sum += float(row[0])
                    sum = sum/LOADED_SONGS

                with open(cv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    file.truncate()
                    writer.writerow([sum])
            else:
                bc_path = os.path.join(folder_name,"bc.csv")
                bd_path = os.path.join(folder_name, "bd.csv")
                bv_path = os.path.join(folder_name,"bv.csv")
                cd_path = os.path.join(folder_name,"cd.csv")
                cv_path = os.path.join(folder_name,"cv.csv")
                dv_path = os.path.join(folder_name,"dv.csv")

                sum = 0
                with open(bc_path) as file:
                    sum = 0
                    for row in csv.reader(file):
                        sum += float(row[0])
                    sum = sum/LOADED_SONGS

                with open(bc_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    file.truncate()
                    writer.writerow([sum])

                with open(bd_path) as file:
                    sum = 0
                    for row in csv.reader(file):
                        sum += float(row[0])
                    sum = sum/LOADED_SONGS

                with open(bd_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    file.truncate()
                    writer.writerow([sum])
                
                with open(bv_path) as file:
                    sum = 0
                    for row in csv.reader(file):
                        sum += float(row[0])
                    sum = sum/LOADED_SONGS

                with open(bv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    file.truncate()
                    writer.writerow([sum])

                with open(cd_path) as file:
                    sum = 0
                    for row in csv.reader(file):
                        sum += float(row[0])
                    sum = sum/LOADED_SONGS

                with open(cd_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    file.truncate()
                    writer.writerow([sum])

                with open(cv_path) as file:
                    sum = 0
                    for row in csv.reader(file):
                        sum += float(row[0])
                    sum = sum/LOADED_SONGS

                with open(cv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    file.truncate()
                    writer.writerow([sum])
                
                with open(dv_path) as file:
                    sum = 0
                    for row in csv.reader(file):
                        sum += float(row[0])
                    sum = sum/LOADED_SONGS

                with open(dv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    file.truncate()
                    writer.writerow([sum])



    



folder_path = os.path.join(os.getcwd(), "songs")
song_folder = os.path.join(folder_path, "song21")

bass_file = os.path.join(song_folder,"bass.mp3")
drums_file = os.path.join(song_folder,"drums.mp3")
chords_file = os.path.join(song_folder,"other.mp3")
vocals_file = os.path.join(song_folder,"vocals.mp3")
        
y_bass, sr_bass = librosa.load(bass_file)
y_drums, sr_drums = librosa.load(drums_file)
y_chords, sr_chords = librosa.load(chords_file)
y_vocals, sr_vocals = librosa.load(vocals_file)

y_min = min(len(y_bass), len(y_drums), len(y_chords), len(y_vocals)) - 1

y_chords = y_chords[0:y_min]
y_drums = y_drums[0:y_min]
y_bass = y_bass[0:y_min]
y_vocals = y_vocals[0:y_min]

bass_stft = librosa_test.stft(y_bass, sr_bass)
drums_stft = librosa_test.stft(y_drums, sr_drums)
chords_stft = librosa_test.stft(y_chords, sr_chords)
vocals_stft = librosa_test.stft(y_vocals, sr_vocals)

bass_chroma = librosa_test.chroma(y_bass, sr_bass)
chords_chroma = librosa_test.chroma(y_chords, sr_chords)
vocals_chroma = librosa_test.chroma(y_vocals, sr_vocals)

bass_rms = librosa_test.rms(y_bass, sr_bass)
drums_rms = librosa_test.rms(y_drums, sr_drums)
chords_rms = librosa_test.rms(y_chords, sr_chords)
vocals_rms = librosa_test.rms(y_vocals, sr_vocals)

bass_centroid = librosa_test.centroid(y_bass, sr_bass)
drums_centroid = librosa_test.centroid(y_drums, sr_drums)
chords_centroid = librosa_test.centroid(y_chords, sr_chords)
vocals_centroid = librosa_test.centroid(y_vocals, sr_vocals)

bass_tonnetz = librosa_test.tonnetz(y_bass, sr_bass, bass_chroma)
chords_tonnetz = librosa_test.tonnetz(y_chords, sr_chords, chords_chroma)
vocals_tonnetz = librosa_test.tonnetz(y_vocals, sr_vocals, vocals_chroma)

add_to_centroid_csv(bass_centroid, drums_centroid, chords_centroid, vocals_centroid)
add_to_chroma_csv(bass_chroma, chords_chroma, vocals_chroma)
add_to_tonnetz_csv(bass_tonnetz, chords_tonnetz, vocals_tonnetz)
add_to_rms_csv(bass_rms, drums_rms, chords_rms, vocals_rms)
add_to_stft_csv(bass_stft, drums_stft, chords_stft, vocals_stft)
