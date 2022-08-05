# provides functions to extract opensmile features for the first and last chunk 
# in every turn of a given corpus (based on normalized low-level descriptors),
# combine them into triplets, and dump them to pickle files in different formats

import itertools
import math
import numpy as np
import os
import pickle
import sqlite3
import subprocess
import time
import torch

import cfg


################################################################################
#                                 PREPARATION                                  #
################################################################################

def set_last_in_turn(corpus_id):
    ''' identify last chunk in every turn of given corpus 
    
    args:
        corpus_id: one of the constants defined in cfg, identifying the corpus
    '''
    cfg.check_corpus_id(corpus_id)
    db_fname = cfg.get_db_fname(corpus_id)
    
    sql_stmt = \
        'UPDATE chunks\n' \
        'SET    is_last_in_turn = 1\n' \
        'WHERE  chu_id IN (\n' \
        '           SELECT chu.chu_id\n' \
        '           FROM   chunks chu\n' \
        '           WHERE  NOT EXISTS (\n' \
        '                      SELECT 1\n' \
        '                      FROM   chunks chu2\n' \
        '                      WHERE  chu.tur_id == chu2.tur_id\n' \
        '                      AND    chu.chunk_index < chu2.chunk_index\n' \
        '                  )\n' \
        '       );'

    with sqlite3.connect(db_fname) as conn:
        c = conn.cursor()
        c.execute(sql_stmt)
        conn.commit()


################################################################################
#                              FEATURE EXTRACTION                              #
################################################################################

def extract(corpus_id, audio_dir, ses_id):
    ''' extract features for all turn-initial/-final chunks of given session 
    
    determines relevant chunks and their timestamps from db, processes their 
    audio with opensmile for low-level descriptors, normalizes the llds, applies
    functionals with opensmile for final features, and writes those to db

    handles database locks so it can be run in parallel
    
    args: 
        corpus_id: one of the constants defined in cfg, identifying the corpus
        audio_dir: audio subdirectory (only for fisher corpus)
        ses_id: session id (normally int, but string with leading 0s for fisher)
    '''
    cfg.check_corpus_id(corpus_id)
    db_fname = cfg.get_db_fname(corpus_id)
    corpus_dir = cfg.get_corpus_dir(corpus_id)
    
    # fisher corpus uses other input columns for the timestamps 
    # (legacy of other code with an additional preprocessing step to remove 
    #  pauses from within transcription segments; this is not done here, so the 
    #  original timestamps are used and copied to standard columns later)
    c1 = 'start_time_orig' if corpus_id == cfg.CORPUS_ID_FISHER \
        else 'start_time'
    c2 = 'end_time_orig' if corpus_id == cfg.CORPUS_ID_FISHER else 'end_time'

    # all turn-initial/-final chunks of given session and speaker
    sql_stmt1 = \
        'SELECT chu.chu_id,\n' \
        '       chu.' + c1 + ',\n' \
        '       chu.' + c2 + '\n' \
        'FROM   chunks chu\n' \
        'JOIN   turns tur\n' \
        'ON     chu.tur_id == tur.tur_id\n' \
        'JOIN   tasks tsk\n' \
        'ON     tur.tsk_id == tsk.tsk_id\n' \
        'WHERE (chu.chunk_index == 1 OR chu.is_last_in_turn == 1)\n' \
        'AND    tsk.ses_id == ?\n' \
        'AND    CASE\n' \
        '           WHEN tur.speaker_role == "d" AND tsk.a_or_b == "A"\n' \
        '           THEN "A"\n' \
        '           WHEN tur.speaker_role == "f" AND tsk.a_or_b == "B"\n' \
        '           THEN "A"\n' \
        '           ELSE "B"\n' \
        '       END == ?\n' \
        'ORDER BY tur.turn_index, chu.chunk_index;'

    sql_stmt2 = \
        'UPDATE chunks\n' \
        'SET    opensmile_features = ?,\n' \
        '       start_time = ?,\n' \
        '       end_time = ?,\n' \
        '       duration = ?\n' \
        'WHERE  chu_id == ?;'

    sql_stmt3 = \
        'UPDATE sessions\n' \
        'SET    status = 2\n' \
        'WHERE  ses_id == ?;'
    
    # convert fisher corpus audio files to wav so opensmile can process them 
    if corpus_id == cfg.CORPUS_ID_FISHER:
        ses_id_str = ses_id
        ses_id = int(ses_id_str)
        sph_fname = '%s%s/audio/%s/fe_03_%s.sph' % \
                    (corpus_dir, audio_dir, ses_id_str[:3], ses_id_str)
        # extract channels, convert to wav with 16 bit signed int pcm
        for a_or_b, chan in [('A', '1'), ('B', '2')]:
            wav_fname = '%s%d_%s.wav' % (cfg.TMP_DIR, ses_id, a_or_b)
            subprocess.check_call([cfg.SPH2PIPE_FNAME, '-f', 'rif', '-p', 
                '-c', chan, sph_fname, wav_fname])
    
    for a_or_b in ['A', 'B']:
        with sqlite3.connect(db_fname) as conn:
            c = conn.cursor()
            c.execute(sql_stmt1, (ses_id, a_or_b))
            if corpus_id == cfg.CORPUS_ID_FISHER:
                in_fname = '%s%d_%s.wav' % (cfg.TMP_DIR, ses_id, a_or_b)
            else:
                in_fname = '%ss%02d.objects.1.%s.wav' % \
                    (corpus_dir, ses_id, a_or_b)
            lld1 = []
            lld2 = []
            chunks = c.fetchall()
            
            for chu_id, start, end in chunks:
                # skip chunks that are too short
                if end - start < 0.1:
                    continue
                
                # determine all necessary file names
                params = (cfg.TMP_DIR, ses_id, a_or_b, chu_id)
                cut_fname = '%s%d_%s_%d.wav' % params
                csv_fname1 = '%s%d_%s_%d.csv' % params
                csv_fname2 = '%s%d_%s_%d_nz.csv' % params
                
                # extract audio section (only stored if intensity ok)
                subprocess.check_call(
                    ['praat', '--run', cfg.PRAAT_FNAME, in_fname, cut_fname,
                     str(start), str(end)])
                
                if os.path.isfile(cut_fname):
                    # run opensmile script to extract low-level descr.
                    subprocess.check_call(
                        ['SMILExtract', '-C', cfg.SMILE_LLD_FNAME, 
                         '-I', cut_fname, '-csvoutput', csv_fname1,
                         '-csvoutput_nz', csv_fname2, '-nologfile'])
        
                    # store all lld for subsequent normalization
                    with open(csv_fname1, 'r') as csv_file1:
                        lld1.extend([l[:-1] for l in csv_file1.readlines()])
                    with open(csv_fname2, 'r') as csv_file2:
                        lld2.extend([l[:-1] for l in csv_file2.readlines()])
                    
                    # clean up
                    os.remove(cut_fname)
            
            # compute mean and standard deviations per lld
            # (1: lld per frame (rows) by splitting csv strings
            #  2: values per lld (cols) across frames (rows) with zip
            #  3: remove 0 values from columns (only for nz lld)
            #  4: mean/std per column (i.e. lld))
            stats1 = [(np.mean(col), np.std(col) if np.std(col) != 0.0 else 1.0) 
                      for col in zip(*[
                          [float(v) for v in csvs.split(';')]
                          for csvs in lld1])]
            stats2 = [(np.mean(col1), np.std(col1) if np.std(col1) != 0.0 
                       else 1.0)
                      for col1 in [[v for v in col0 if v != 0.0] 
                                   for col0 in zip(*[
                                       [float(v) for v in csvs.split(';')]
                                       for csvs in lld2])]]
            
            # determine whether any of the stats are nan
            # (can happen as result of 0 for a feature across all chunks;
            #  such recordings do not seem reliable so they are ignored)
            if len(stats1) == 0 or len(stats2) == 0:
                any_nan = True
            else:
                any_nan = max([math.isnan(v[0]) or math.isnan(v[1]) 
                               for v in stats1]) \
                       or max([math.isnan(v[0]) or math.isnan(v[1]) 
                               for v in stats2])

            # normalize, apply funcs to lld of each chunk to get features
            for chu_id, start, end in chunks:
                params = (cfg.TMP_DIR, ses_id, a_or_b, chu_id)
                csv_fname1 = '%s%d_%s_%d.csv' % params
                csv_fname2 = '%s%d_%s_%d_nz.csv' % params
                out_fname = '%s%d_%s_%d_out.csv' % params
                
                if not os.path.isfile(csv_fname1):
                    continue
                
                # only compute features if all lld stats are not nan
                if not any_nan:
                    # read lld output from above
                    with open(csv_fname1, 'r') as csv_file1:
                        lld1 = [l[:-1] for l in csv_file1.readlines()]
                    with open(csv_fname2, 'r') as csv_file2:
                        lld2 = [l[:-1] for l in csv_file2.readlines()]

                    # compute normalized lld
                    lld_nrm1 = [[str((v - stats1[i][0]) / stats1[i][1])
                                 for i, v in enumerate(row)] for row in 
                                    [[float(v) for v in csvs.split(';')] 
                                     for csvs in lld1]]
                    lld_nrm2 = [[str((v - stats2[i][0]) / stats2[i][1])
                                 if v != 0.0 else '0.0' 
                                 for i, v in enumerate(row)] for row in 
                                    [[float(v) for v in csvs.split(';')] 
                                     for csvs in lld2]]

                    # overwrite files with normalized lld 
                    with open(csv_fname1, 'w') as csv_file1:
                        csv_file1.write('\n'.join([';'.join(row) 
                                                   for row in lld_nrm1]))
                    with open(csv_fname2, 'w') as csv_file2:
                        csv_file2.write('\n'.join([';'.join(row) 
                                                   for row in lld_nrm2]))

                    # run opensmile script to apply functions to get features
                    subprocess.check_call(
                        ['SMILExtract', '-C', cfg.SMILE_FUN_FNAME, 
                         '-I', csv_fname1, '-J', csv_fname2,
                         '-csvoutput', out_fname, '-nologfile'])
                # clean up
                os.remove(csv_fname1)
                os.remove(csv_fname2)
                
            done = False
            if not any_nan:
                while not done:
                    try:
                        for chu_id, start, end in chunks:
                            params = (cfg.TMP_DIR, ses_id, a_or_b, chu_id)
                            out_fname = '%s%d_%s_%d_out.csv' % params
                            if not os.path.isfile(out_fname):
                                continue
                            # write features to db
                            with open(out_fname, 'r') as out_file:
                                c.execute(
                                    sql_stmt2, 
                                    (out_file.readlines()[0], 
                                     start, end, end - start, chu_id))
                            os.remove(out_fname)
                        if corpus_id == cfg.CORPUS_ID_FISHER:
                            c.execute(sql_stmt3, (ses_id,))
                        conn.commit()
                        done = True
                    except sqlite3.OperationalError:
                        pass
    
    # clean up wav files
    if corpus_id == cfg.CORPUS_ID_FISHER:
        for a_or_b, chan in [('A', '1'), ('B', '2')]:
            wav_fname = '%s%d_%s.wav' % (cfg.TMP_DIR, ses_id, a_or_b)
            os.remove(wav_fname)


################################################################################
#                               TRIPLET CREATION                               #
################################################################################

def load_triplets(corpus_id, print_status=False):
    ''' forms turn exchange chunk triplets and loads their opensmile features

    args:
        corpus_id: one of the constants defined in cfg, identifying the corpus
        print_status: whether to print regular status updates or not
    returns:
        dict mapping task/session id to two lists of triplets, one per 
        speaker ('A', 'B')
    '''
    cfg.check_corpus_id(corpus_id)
    db_fname = cfg.get_db_fname(corpus_id)
    
    if corpus_id == cfg.CORPUS_ID_FISHER:
        sql_stmt_sel_tsk_ses = \
            'SELECT ses_id\n' \
            'FROM   sessions\n' \
            'WHERE  status == 2\n' \
            'ORDER BY ses_id;'
    else:
        sql_stmt_sel_tsk_ses = \
            'SELECT tsk_id FROM tasks ORDER BY tsk_id;'

    # code uses temporary tables created and dropped for each task or session;
    # table for first and last chunks in every turn of one task or session
    sql_stmt_cre_chu = \
        'CREATE TABLE chu (\n' \
        '    a_or_b                TEXT,\n' \
        '    turn_index            INTEGER,\n' \
        '    turn_type             TEXT,\n' \
        '    chunk_index           INTEGER,\n' \
        '    is_last_in_turn       INTEGER,\n' \
        '    opensmile_features    TEXT\n' \
        ');'

    # table for first chunks in consecutive turns of one speaker in table chu
    sql_stmt_cre_chu1_3 = \
        'CREATE TABLE chu1_3 (\n' \
        '    a_or_b         TEXT,\n' \
        '    turn_index1    INTEGER,\n' \
        '    features1      TEXT,\n' \
        '    turn_index3    INTEGER,\n' \
        '    features3      TEXT\n' \
        ');'

    col = 'tsk.ses_id' if corpus_id == cfg.CORPUS_ID_FISHER else 'tur.tsk_id'
    # populate table chu for given task or session
    sql_stmt_ins_chu = \
        'INSERT INTO chu\n' \
        'SELECT CASE\n' \
        '           WHEN tur.speaker_role == "d" AND tsk.a_or_b == "A"\n' \
        '           THEN "A"\n' \
        '           WHEN tur.speaker_role == "f" AND tsk.a_or_b == "B"\n' \
        '           THEN "A"\n' \
        '           ELSE "B"\n' \
        '       END a_or_b,\n' \
        '       tur.turn_index,\n' \
        '       tur.turn_type,\n' \
        '       chu.chunk_index,\n' \
        '       chu.is_last_in_turn,\n' \
        '       chu.opensmile_features\n' \
        'FROM   chunks chu\n' \
        'JOIN   turns tur\n' \
        'ON     chu.tur_id == tur.tur_id\n' \
        'JOIN   tasks tsk\n' \
        'ON     tur.tsk_id == tsk.tsk_id\n' \
        'WHERE (chu.chunk_index == 1 OR chu.is_last_in_turn == 1)\n' \
        'AND    chu.opensmile_features IS NOT NULL\n' \
        'AND    ' + col + ' == ?;'

    # populate table chu1_3 for given speaker from table chu
    sql_stmt_ins_chu1_3 = \
        'INSERT INTO chu1_3\n' \
        'SELECT chu1.a_or_b,\n' \
        '       chu1.turn_index turn_index1,\n' \
        '       chu1.opensmile_features features1,\n' \
        '       chu3.turn_index turn_index3,\n' \
        '       chu3.opensmile_features features3\n' \
        'FROM   chu chu1\n' \
        'JOIN   chu chu3\n' \
        'ON     chu1.a_or_b == chu3.a_or_b\n' \
        'WHERE  chu1.a_or_b == ?\n' \
        'AND    chu1.chunk_index == 1\n' \
        'AND    chu3.chunk_index == 1\n' \
        'AND    chu3.turn_index == (\n' \
        '                       SELECT MIN(turn_index)\n' \
        '                       FROM   chu chu4\n' \
        '                       WHERE  chu4.a_or_b == chu1.a_or_b\n' \
        '                       AND    chu4.turn_index > chu1.turn_index\n' \
        '                      );'

    # construct triplets from tables chu and chu1_3
    sql_stmt_get_chu123 = \
        'SELECT chu1_3.features1,\n' \
        '       chu2.opensmile_features features2, \n' \
        '       chu1_3.features3\n' \
        'FROM   chu1_3\n' \
        'JOIN   chu chu2\n' \
        'ON     chu1_3.a_or_b != chu2.a_or_b\n' \
        'WHERE  chu2.is_last_in_turn == 1\n' \
        'AND    chu2.turn_index == (\n' \
        '                   SELECT MAX(turn_index)\n' \
        '                   FROM   chu chu4\n' \
        '                   WHERE  chu4.a_or_b != chu1_3.a_or_b\n' \
        '                   AND    chu4.turn_index > chu1_3.turn_index1\n' \
        '                   AND    chu4.turn_index < chu1_3.turn_index3\n' \
        '                  )\n' \
        'ORDER BY chu1_3.turn_index1;'

    # actually execute the sql statements above to load triplets
    with sqlite3.connect(db_fname) as conn:
        c1 = conn.cursor()
        c2 = conn.cursor()
        
        # create the tmp tables
        c1.execute('DROP TABLE IF EXISTS chu;')
        c1.execute('DROP TABLE IF EXISTS chu1_3;')
        c1.execute(sql_stmt_cre_chu)
        c1.execute(sql_stmt_cre_chu1_3)
        conn.commit()
        
        # extract triplets per task/session and speaker
        data = {}
        c1.execute(sql_stmt_sel_tsk_ses)
        tsk_ses_ids = [v[0] for v in c1.fetchall()]
        for i, tsk_ses_id in enumerate(tsk_ses_ids):
            if print_status and i % int(len(tsk_ses_ids)/10) == 0:
                print('%d%% done %s' % 
                      (i/int(len(tsk_ses_ids)/10)*10, time.ctime()))
            data[tsk_ses_id] = {}
            c1.execute(sql_stmt_ins_chu, (tsk_ses_id,))
            for a_or_b in ['A', 'B']:
                c1.execute(sql_stmt_ins_chu1_3, (a_or_b,))
                c2.execute(sql_stmt_get_chu123)
                # parse list of triplets of csv feature lists to float tensor
                data[tsk_ses_id][a_or_b] = \
                    torch.tensor([[[float(v) for v in csv.split(';')]
                                   for csv in csv_triplet] 
                                  for csv_triplet in c2.fetchall()])
                c2.execute('DELETE FROM chu1_3;')
            c2.execute('DELETE FROM chu;')
            # remove sessions with low number of triplets 
            # (often the result of transcription issues)
            if corpus_id == cfg.CORPUS_ID_FISHER \
            and (len(data[tsk_ses_id]['A']) < 11 
              or len(data[tsk_ses_id]['B']) < 11):
                del data[tsk_ses_id]
        c1.execute('DROP TABLE IF EXISTS chu;')
        c1.execute('DROP TABLE IF EXISTS chu1_3;')
        conn.commit()

    return data
        

################################################################################
#                               PICKLE FUNCTIONS                               #
################################################################################

def pickle_dump(corpus_id, data_id, with_ipu0, data):
    ''' stores given data in pickle file, in filename for given metadata 

    intended for triplets but technically the data can be in any format 

    args:
        corpus_id: one of the constants defined in cfg, identifying the corpus
        data_id: one of the constants defined in cfg, identifying data format
        with_ipu0: whether data has been transformed to ipu0 triplets (1) or 
            not (0); see change_to_data0 
        data: data in any format
    '''
    pickle_fname = cfg.get_pickle_fname(corpus_id, data_id, with_ipu0)
    with open(pickle_fname, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
    print('...done! %s' % time.ctime())


def pickle_load(corpus_id, data_id, with_ipu0):
    ''' loads data from pickle file with name based on given metadata
    
    args:
        corpus_id: one of the constants defined in cfg, identifying the corpus
        data_id: one of the constants defined in cfg, identifying data format
        with_ipu0: whether to load data with ipu0 triplets (1) or not (0)
    '''
    pickle_fname = cfg.get_pickle_fname(corpus_id, data_id, with_ipu0)
    with open(pickle_fname, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    print('...done! %s' % time.ctime())
    return data


################################################################################
#                               FEATURE STORAGE                                #
################################################################################

def get_data_per_set_and_tsk_or_ses(data_in):
    ''' returns data split into 80/10/10 subsets, maintaining task/session info
    
    percentage split is based on task/session count, not triplet count

    args:
        data_in: data as returned by load_triplets, two tensors per task/session
    returns:
        dict mapping subset id to dict mapping session id to feature tensors,
        one per speaker ('A', 'B')
    '''
    data_out = {'trn': {}, 'vld': {}, 'tst': {}}
    data_out['trn'] = {key: val for i, (key, val) in enumerate(data_in.items())
                       if i < 0.8 * len(data_in)}
    data_out['vld'] = {key: val for i, (key, val) in enumerate(data_in.items())
                       if 0.8 * len(data_in) <= i and i < 0.9 * len(data_in)}
    data_out['tst'] = {key: val for i, (key, val) in enumerate(data_in.items())
                       if 0.9 * len(data_in) <= i}
    return data_out


def get_data_per_set(data_in):
    ''' returns data split into subsets, removing task/session info from input
    
    percentage split based on input, stays unchanged

    args:
        data_in: data as returned by get_data_per_set_and_tsk_or_ses
    returns:
        dict mapping subset id to tensor with features of all speakers
    '''
    return {
        idx: torch.cat([data_in[idx][tsk_ses_id][a_or_b]
                        for tsk_ses_id, a_or_b in itertools.product(
                            data_in[idx].keys(), ['A', 'B'])],
                       0)
        for idx in ['trn', 'vld', 'tst']
    }


def get_data_per_gc_ses(data_in):
    ''' returns data with session info, based on data with task info 

    assumes games corpus specifics: 12 sessions with 14 tasks each

    args:
        data_in: data per task, as returned by load_triplets for games corpus
    returns:
        dict mapping session id to two tensors, one per speaker ('A', 'B'),
        with torch.Size([N, 3, 228]) where N is a varying number of triplets
    '''
    # note: code assumes specifics of games corpus (12 ses, 14 tsk each)
    data_out = {(ses_id+1): {} for ses_id in range(12)}
    
    for ses_id in range(12):
        for a_or_b in ['A', 'B']:
            data_out[ses_id+1][a_or_b] = torch.cat(
                [data_in[ses_id*14 + tsk_id][a_or_b]
                 for tsk_id in range(1, 15)],
                0)
    
    return data_out


def change_to_data0(corpus_id, data):
    ''' replaces 1st ipu per triplet in input data by speaker's 1st in session 

    args:
        corpus_id: one of the constants defined in cfg, identifying the corpus
        data_in: data as returned by load_triplets, i.e., triplets 
            per task (games corpus) or session (fisher)
    '''
    cfg.check_corpus_id(corpus_id)
    # note: code assumes 14 tasks per session (true for games corpus)
    ipu0 = None
    for tsk_ses_id, speaker_triplets in data.items():
        for a_or_b in ['A', 'B']:
            if len(speaker_triplets[a_or_b]) == 0:
                # skip tasks/sessions without data 
                # (these had nan features in opensmile, see extract)
                continue
            if corpus_id == cfg.CORPUS_ID_FISHER:
                ipu0 = speaker_triplets[a_or_b][0][0]
            else:
                if ipu0 is None:
                    ipu0 = speaker_triplets[a_or_b][0][0]
            for i in range(1, len(speaker_triplets[a_or_b])):
                speaker_triplets[a_or_b][i][0] = ipu0.clone().detach()
            if corpus_id == cfg.CORPUS_ID_GAMES and tsk_ses_id % 14 == 0:
                ipu0 = None
    

def store_fisher_data(data, with_ipu0):
    ''' stores data for fisher corpus in different formats

    args:
        data: data per session, as returned by load_triplets for fisher corpus
        with_ipu0: whether data has been transformed to ipu0 triplets (1) or 
            not (0); see change_to_data0 
    '''
    print('storing data per session...')
    pickle_dump(cfg.CORPUS_ID_FISHER, cfg.DATA_ID_SES, with_ipu0, data)

    print('storing data per set (trn, dev, tst) and session...')
    data_per_set_and_ses = get_data_per_set_and_tsk_or_ses(data)
    pickle_dump(cfg.CORPUS_ID_FISHER, cfg.DATA_ID_SET_SES, with_ipu0, 
         data_per_set_and_ses)

    print('storing data per set (trn, dev, tst)...')
    data_per_set = get_data_per_set(data_per_set_and_ses)
    pickle_dump(cfg.CORPUS_ID_FISHER, cfg.DATA_ID_SET, with_ipu0, data_per_set)


def store_games_data(data, with_ipu0):
    ''' stores data for games corpus in different formats

    args:
        data: data per task, as returned by load_triplets for games corpus
        with_ipu0: whether data has been transformed to ipu0 triplets (1) or 
            not (0); see change_to_data0
    '''
    print('storing data per task...')
    pickle_dump(cfg.CORPUS_ID_GAMES, cfg.DATA_ID_TSK, with_ipu0, data)

    print('storing data per session...')
    data_per_ses = get_data_per_gc_ses(data)
    pickle_dump(cfg.CORPUS_ID_GAMES, cfg.DATA_ID_SES, with_ipu0, data_per_ses)


def load_and_store_triplets(corpus_id, print_status=False):
    ''' combines all aux functions to compute and store data as needed 

    args:
        corpus_id: one of the constants defined in cfg, identifying the corpus
        print_status: whether to print regular status updates or not
    '''
    cfg.check_corpus_id(corpus_id)

    print('loading data for all tasks/sessions...')
    data = load_triplets(corpus_id, print_status)
    print('...done! %s\n' % time.ctime())

    print(sum([len(data[key]['A']) + len(data[key]['B']) 
               for key, val in data.items()]), 'triplets total\n')

    if corpus_id == cfg.CORPUS_ID_FISHER:
        store_fisher_data(data, 0)
        print('\nconverting to triplets with IPU 0...')
        change_to_data0(corpus_id, data)
        print('...done! %s\n' % time.ctime())
        store_fisher_data(data, 1)
    else:
        store_games_data(data, 0)
        print('\nconverting to triplets with IPU 0...')
        change_to_data0(corpus_id, data)
        print('...done! %s\n' % time.ctime())
        store_games_data(data, 1)



