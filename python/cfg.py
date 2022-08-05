SMILE_LLD_FNAME = '../smile/lld.conf'
SMILE_FUN_FNAME = '../smile/fun.conf'
PRAAT_FNAME = '../misc/extract_part.praat'
SPH2PIPE_FNAME = '../misc/sph2pipe'

# set as needed
EXT_DIR = ''
PICKLE_DIR = '%scorpora/smile_features/' % EXT_DIR
TMP_DIR = '%stmp/' % EXT_DIR
DUMPS_DIR = '../../dumps/'
NETS_DIR = '%snets/' % DUMPS_DIR

DB_FNAME_GC = '../../gc.db'
CORPUS_DIR_GC = '%scorpora/games_corpus/' % EXT_DIR

DB_FNAME_FC = '../../fc.db'
CORPUS_DIR_FC = '%scorpora/fisher_corpus/' % EXT_DIR
FILELIST_FNAME_FC = '%s/meta/fe_03_p%%d_filelist.tbl' % CORPUS_DIR_FC

SES_TEST_RES_FNAME = '%sfake_session_test_results.pickle' % DUMPS_DIR
COR_TEST_RES_FNAME = '%scorrelation_test_results.pickle' % DUMPS_DIR

CORPUS_ID_FISHER = 'FC'
CORPUS_ID_GAMES  = 'GC'
CORPUS_IDS = [CORPUS_ID_FISHER, CORPUS_ID_GAMES]

MEASURE_ID_DR = 'DR'
MEASURE_ID_AS = 'AS'
MEASURE_IDS = [MEASURE_ID_DR, MEASURE_ID_AS]

IN_SIZE = 228 # number of features, standard input size for networks
HIDDEN_SIZE1 = 128
HIDDEN_SIZE2 = 30
BATCH_SIZE = 128

DATA_ID_SES = 'ses'
DATA_ID_SET = 'set'
DATA_ID_SET_SES = 'set_ses'
DATA_ID_TSK = 'tsk'

PICKLE_FNAMES = {
    CORPUS_ID_FISHER: {
        DATA_ID_SES: [
            '%sfc_triplets_per_ses.pickle' % PICKLE_DIR,
            '%sfc_triplets_with_ipu0_per_ses.pickle' % PICKLE_DIR
        ],
        DATA_ID_SET: [
            '%sfc_triplets_per_set.pickle' % PICKLE_DIR,
            '%sfc_triplets_with_ipu0_per_set.pickle' % PICKLE_DIR
        ],
        DATA_ID_SET_SES: [
            '%sfc_triplets_per_set_and_ses.pickle' % PICKLE_DIR,
            '%sfc_triplets_with_ipu0_per_set_and_ses.pickle' % PICKLE_DIR
        ]
    },
    CORPUS_ID_GAMES: {
        DATA_ID_SES: [
            '%sgc_triplets_per_ses.pickle' % PICKLE_DIR,
            '%sgc_triplets_with_ipu0_per_ses.pickle' % PICKLE_DIR
        ],   
        DATA_ID_TSK: [
            '%sgc_triplets_per_tsk.pickle' % PICKLE_DIR,
            '%sgc_triplets_with_ipu0_per_tsk.pickle' % PICKLE_DIR
        ]
    }
}

SOCIAL_VARS = [
    ('enc', 'gives_encouragement'),
    ('tli', 'trying_to_be_liked'),
    ('tdo', 'trying_to_dominate')
]


def check_corpus_id(corpus_id):
    assert corpus_id in [CORPUS_ID_FISHER, CORPUS_ID_GAMES], 'unknown corpus id'


def get_db_fname(corpus_id):
    check_corpus_id(corpus_id)
    return DB_FNAME_FC if corpus_id == CORPUS_ID_FISHER else DB_FNAME_GC


def get_corpus_dir(corpus_id):
    check_corpus_id(corpus_id)
    return CORPUS_DIR_FC if corpus_id == CORPUS_ID_FISHER else CORPUS_DIR_GC


def get_pickle_fname(corpus_id, data_id, with_ipu0):
    return PICKLE_FNAMES[corpus_id][data_id][with_ipu0]







