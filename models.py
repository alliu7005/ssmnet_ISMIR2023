from flask_sqlalchemy import SQLAlchemy
from song_struct import Song_Struct, Stem
import numpy as np


db = SQLAlchemy()

class SongModel(db.Model):
    __tablename__ = 'songs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    sr = db.Column(db.Integer)
    tempo = db.Column(db.Float)
    key = db.Column(db.Integer)
    major = db.Column(db.Integer)
    bounds = db.Column(db.JSON)
    downbeats = db.Column(db.PickleType)
    beats = db.Column(db.PickleType)
    y = db.Column(db.PickleType)
    stems = db.relationship("StemModel", back_populates="song", cascade="all, delete-orphan")

class StemModel(db.Model):
    __tablename__ = 'stems'

    id = db.Column(db.Integer, primary_key=True)
    song_id = db.Column(db.Integer, db.ForeignKey('songs.id'))
    name = db.Column(db.String)
    songname = db.Column(db.String)
    sr = db.Column(db.Integer)
    tempo = db.Column(db.Float)
    key = db.Column(db.Integer)
    major = db.Column(db.Integer)
    bounds = db.Column(db.JSON)
    downbeats = db.Column(db.PickleType)
    beats = db.Column(db.PickleType)
    y = db.Column(db.PickleType)
    silent = db.Column(db.PickleType)
    active = db.Column(db.PickleType)
    chroma = db.Column(db.PickleType)
    mfcc = db.Column(db.PickleType)
    stft = db.Column(db.PickleType)
    rms = db.Column(db.PickleType)
    tonnetz = db.Column(db.PickleType)
    specgram = db.Column(db.PickleType)
    song = db.relationship("SongModel", back_populates="stems")

class GraphModel(db.Model):
    __tablename__ = 'graph'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    data = db.Column(db.JSON)

def song_from_orm(orm):
    name = orm.name
    sr = orm.sr
    orig_y = orm.y
    tempo = orm.tempo
    key = orm.key
    major = orm.major
    bounds = orm.bounds
    downbeats = np.around(orm.downbeats,2)
    beats = np.around(orm.beats,2)
    stems = orm.stems
    for stem in stems:
        if stem.name == "vocals":
            v_y = stem.y
        elif stem.name == "other":
            o_y = stem.y
        elif stem.name == "bass":
            b_y = stem.y
        else:
            d_y = stem.y

    song = Song_Struct(orig_y, sr, name, v_y=v_y, o_y=o_y, b_y=b_y, d_y=d_y, tempo=tempo, key=key, major=major, bounds=bounds, downbeats=downbeats, beats=beats, take_fields=False)

    for i in range(len(song.stems)):
        song.stems[i].active = stems[i].active
        song.stems[i].silence = stems[i].silent
        song.stems[i].chroma = stems[i].chroma
        song.stems[i].rms = stems[i].rms
        song.stems[i].tonnetz = stems[i].tonnetz
        song.stems[i].specgram = stems[i].specgram
        song.stems[i].mfcc = stems[i].mfcc
        song.stems[i].stft = stems[i].stft

    return song

def stem_from_orm(orm):
    init_song = song_from_orm(orm.song)
    y = orm.y
    sr = orm.sr
    name = orm.name
    tempo = orm.tempo
    key = orm.key
    major = orm.major
    bounds = orm.bounds
    downbeats = orm.downbeats
    beats = orm.beats
    silence = orm.silent
    active = orm.active
    chroma = orm.chroma
    stft = orm.stft
    rms = orm.rms
    tonnetz = orm.tonnetz
    mfcc = orm.mfcc
    specgram = orm.specgram
    
    stem = Stem(
        init_song=init_song,
        y=y,
        sr=sr,
        name=name,
        tempo=tempo,
        key=key,
        major=major,
        bounds=bounds,
        downbeats=downbeats,
        beats=beats,
        silence = silence,
        active = active,
        chroma = chroma,
        stft = stft,
        rms = rms,
        tonnetz = tonnetz,
        mfcc = mfcc,
        specgram = specgram
    )
    return stem