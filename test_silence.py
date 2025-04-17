import os
import librosa
import librosa_test
import csv
import numpy as np

SONG_FOLDER = "songs/song2/"

UPLOAD_PATH = "silence_features/"


vocals = os.path.join(SONG_FOLDER, "vocals" + ".mp3")
drums = os.path.join(SONG_FOLDER, "drums" + ".mp3")
other = os.path.join(SONG_FOLDER, "other" + ".mp3")
bass = os.path.join(SONG_FOLDER, "bass" + ".mp3")
orig = os.path.join(SONG_FOLDER, "song2.mp3")

v_y, v_sr = librosa.load(vocals)
d_y, d_sr = librosa.load(drums)
c_y, c_sr = librosa.load(other)
b_y, b_sr = librosa.load(bass)



y_dict = {"vocals": v_y,
          "drums": d_y,
          "other": c_y,
          "bass": b_y}

o_y, o_sr = librosa.load(orig)

o_y, index = librosa_test.trim(o_y)



beats, downbeats = librosa_test.get_beats_and_downbeats(o_y)

downbeats_sample = librosa_test.times_to_samples(downbeats, o_sr)
frame_length = int(np.mean(np.diff(downbeats_sample)))

for stem in y_dict:

    path = os.path.join(UPLOAD_PATH, stem + ".csv")

    titlesrow = ["silence start time", "silence end time", "silence start bar", "silence end bar"]

    other_stems = [other for other in y_dict if other != stem]

    if stem == "vocals":
        titlesrow.extend(["bc rms", "bd rms", "cd rms"])
        titlesrow.extend(["bc mfcc", "bd mfcc", "cd mfcc"])
    elif stem == "drums":
        titlesrow.extend(["bc rms", "bv rms", "cv rms"])
        titlesrow.extend(["bc mfcc", "bv mfcc", "cv mfcc"])
    elif stem == "other":
        titlesrow.extend(["bd rms", "bv rms", "dv rms"])
        titlesrow.extend(["bd mfcc", "bv mfcc", "dv mfcc"])
    else:
        titlesrow.extend(["cd rms", "cv rms", "dv rms"])
        titlesrow.extend(["cd mfcc", "cv mfcc", "dv mfcc"])

    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        file.truncate()
        writer.writerow(titlesrow)

    y_dict[stem] = y_dict[stem][index[0]:index[1]]
    intervals = librosa_test.find_silence(y_dict[stem], frame_length)
    #for interval in intervals:
        #print(stem, librosa_test.samples_to_time(interval))

    downbeat_intervals = librosa_test.find_silent_downbeat_ranges(o_sr, intervals, downbeats)
    #for interval in downbeat_intervals:
        #print(stem + "downbeats", librosa_test.samples_to_time(interval))
    #np.around(downbeat_intervals, 2)
    
    for interval in downbeat_intervals:
        time_interval = librosa_test.samples_to_time(interval)
        time_interval = np.around(time_interval, 2)
        downbeats_of_interval = np.array([downbeat for downbeat in downbeats if downbeat >= time_interval[0] and downbeat <= time_interval[1]])
        #np.append(downbeats_of_interval, [downbeat for downbeat in downbeats if downbeat > time_interval[1]][0])
        #print([downbeat for downbeat in downbeats if downbeat > time_interval[1]][0])
        #print(downbeats)
        #print(downbeats_of_interval)
        start_bar = np.where(downbeats == downbeats_of_interval[0])[0][0]
        end_bar = np.where(downbeats == downbeats_of_interval[-1])[0][0]
        row_to_add = [time_interval[0], time_interval[1], start_bar, end_bar]
        
        if stem == "vocals":
            y_drums = y_dict["drums"][interval[0]:interval[1]]
            
            y_chords = y_dict["other"][interval[0]:interval[1]]
            

            y_bass = y_dict["bass"][interval[0]:interval[1]]

            #print(y_drums)
            #print(y_chords)
            #print(y_bass)
            drums_rms = librosa_test.rms(y_drums, o_sr)
            drums_mfcc = librosa_test.mfcc(d_y, o_sr, downbeats_of_interval)

            #print("drumslen", drums_rms.shape)

            chords_rms = librosa_test.rms(y_chords, o_sr)
            chords_mfcc = librosa_test.mfcc(c_y, o_sr, downbeats_of_interval)

            #print("chordslen", chords_rms.shape)

            bass_rms = librosa_test.rms(y_bass, o_sr)
            bass_mfcc = librosa_test.mfcc(b_y, o_sr, downbeats_of_interval)

            #print("basslen", bass_rms.shape)

            bc_rms = librosa_test.corr_coef_split(bass_rms, chords_rms)[0][0]
            bc_mfcc = np.mean(librosa_test.bar_by_bar_corr_coef_mfcc(bass_mfcc, chords_mfcc)[0])

            bd_rms = librosa_test.corr_coef_split(bass_rms, drums_rms)[0][0]
            bd_mfcc = np.mean(librosa_test.bar_by_bar_corr_coef_mfcc(bass_mfcc, drums_mfcc)[0])

            cd_rms = librosa_test.corr_coef_split(chords_rms, drums_rms)[0][0]
            cd_mfcc = np.mean(librosa_test.bar_by_bar_corr_coef_mfcc(chords_mfcc, drums_mfcc)[0])

            #(chords_mfcc)
            #print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            #print(bass_mfcc)

            row_to_add.extend([bc_rms, bd_rms, cd_rms, bc_mfcc, bd_mfcc, cd_mfcc])

        elif stem == "drums":
            y_vocals = y_dict["vocals"][interval[0]:interval[1]]
            y_chords = y_dict["other"][interval[0]:interval[1]]
            y_bass = y_dict["bass"][interval[0]:interval[1]]

            #print(y_vocals)
            #print(y_chords)
            
            #print(y_bass)

            vocals_rms = librosa_test.rms(y_vocals, o_sr)
            vocals_mfcc = librosa_test.mfcc(v_y, o_sr, downbeats_of_interval)

            chords_rms = librosa_test.rms(y_chords, o_sr)
            chords_mfcc = librosa_test.mfcc(c_y, o_sr, downbeats_of_interval)

            bass_rms = librosa_test.rms(y_bass, o_sr)
            bass_mfcc = librosa_test.mfcc(b_y, o_sr, downbeats_of_interval)

            bc_rms = librosa_test.corr_coef_split(bass_rms, chords_rms)[0][0]
            #print(librosa_test.bar_by_bar_corr_coef_mfcc(bass_mfcc, chords_mfcc))

            #print(bass_mfcc, chords_mfcc)
            #print("all intervals", librosa_test.samples_to_time(downbeat_intervals))
            #print(time_interval)
            #print("interval", downbeats_of_interval)
            bc_mfcc = np.mean(librosa_test.bar_by_bar_corr_coef_mfcc(bass_mfcc, chords_mfcc)[0])

            bv_rms = librosa_test.corr_coef_split(bass_rms, vocals_rms)[0][0]
            bv_mfcc = np.mean(librosa_test.bar_by_bar_corr_coef_mfcc(bass_mfcc, vocals_mfcc)[0])

            cv_rms = librosa_test.corr_coef_split(chords_rms, vocals_rms)[0][0]
            cv_mfcc = np.mean(librosa_test.bar_by_bar_corr_coef_mfcc(chords_mfcc, vocals_mfcc)[0])

            row_to_add.extend([bc_rms, bv_rms, cv_rms, bc_mfcc, bv_mfcc, cv_mfcc])

        elif stem == "other":
            y_drums = y_dict["drums"][interval[0]:interval[1]]
            y_vocals = y_dict["vocals"][interval[0]:interval[1]]
            y_bass = y_dict["bass"][interval[0]:interval[1]]

            #print(y_vocals)
            #print(y_drums)
            #print(y_bass)

            vocals_rms = librosa_test.rms(y_vocals, o_sr)
            vocals_mfcc = librosa_test.mfcc(v_y, o_sr, downbeats_of_interval)

            drums_rms = librosa_test.rms(y_drums, o_sr)
            drums_mfcc = librosa_test.mfcc(d_y, o_sr, downbeats_of_interval)

            bass_rms = librosa_test.rms(y_bass, o_sr)
            bass_mfcc = librosa_test.mfcc(b_y, o_sr, downbeats_of_interval)

            bd_rms = librosa_test.corr_coef_split(bass_rms, drums_rms)[0][0]
            bd_mfcc = np.mean(librosa_test.bar_by_bar_corr_coef_mfcc(bass_mfcc, drums_mfcc)[0])

            bv_rms = librosa_test.corr_coef_split(bass_rms, vocals_rms)[0][0]
            bv_mfcc = np.mean(librosa_test.bar_by_bar_corr_coef_mfcc(bass_mfcc, vocals_mfcc)[0])

            dv_rms = librosa_test.corr_coef_split(drums_rms, vocals_rms)[0][0]
            dv_mfcc = np.mean(librosa_test.bar_by_bar_corr_coef_mfcc(drums_mfcc, vocals_mfcc)[0])

            row_to_add.extend([bd_rms, bv_rms, dv_rms, bd_mfcc, bv_mfcc, dv_mfcc])
        else:
            y_drums = y_dict["drums"][interval[0]:interval[1]]
            y_vocals = y_dict["vocals"][interval[0]:interval[1]]
            y_chords = y_dict["other"][interval[0]:interval[1]]

            vocals_rms = librosa_test.rms(y_vocals, o_sr)
            vocals_mfcc = librosa_test.mfcc(v_y, o_sr, downbeats_of_interval)

            drums_rms = librosa_test.rms(y_drums, o_sr)
            drums_mfcc = librosa_test.mfcc(d_y, o_sr, downbeats_of_interval)

            chords_rms = librosa_test.rms(y_chords, o_sr)
            chords_mfcc = librosa_test.mfcc(c_y, o_sr, downbeats_of_interval)

            cd_rms = librosa_test.corr_coef_split(chords_rms, drums_rms)[0][0]
            cd_mfcc = np.mean(librosa_test.bar_by_bar_corr_coef_mfcc(chords_mfcc, drums_mfcc))

            cv_rms = librosa_test.corr_coef_split(chords_rms, vocals_rms)[0][0]
            cv_mfcc = np.mean(librosa_test.bar_by_bar_corr_coef_mfcc(chords_mfcc, vocals_mfcc))

            dv_rms = librosa_test.corr_coef_split(drums_rms, vocals_rms)[0][0]
            dv_mfcc = np.mean(librosa_test.bar_by_bar_corr_coef_mfcc(drums_mfcc, vocals_mfcc))

            row_to_add.extend([cd_rms, cv_rms, dv_rms, cd_mfcc, cv_mfcc, dv_mfcc])

            
            
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row_to_add)
