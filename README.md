# Neural Entrainment Measures

This is the code for the following paper (with slightly different results):
Weise, A., & Levitan, R. (2020). Decoupling entrainment from consistency using deep neural networks. http://arxiv.org/abs/2011.01860

Two decounfounding neural network architectures adapted from Pryzant et al. (2018) are defined and trained on the Fisher Corpus (data not included), then evaluated with a fake session detection test and a check for correlations with social variables of the Columbia Games Corpus (data not included).

Network inputs consist of triplets of feature vectors for a turn exchange (IPU1, last inter-pausal unit / utterance before end of turn and IPU2, first utterance after) and IPU0, the very first utterance of the same speaker as IPU2. Networks consist of encoder and decoder components. See the paper for more details.

## Directory Overview

<ul>
    <li>jupyter: a sequence of Jupyter notebooks that invoke all SQL/python code to process and analyze the corpora</li>
    <li>praat: single Praat script for extraction of audio segments</li>
    <li>python: modules for data processing and analysis invoked from the Jupyter notebooks; file overview:
        <ul>
            <li>cfg.py: configuration constants; if you received the corpus data (separately), configure the correct paths here</li>
            <li>dnn.py: training and testing code for the neural networks</li>
            <li>fea.py: feature extraction code</li>
        </ul>
    </li>
    <li>smile: openSMILE scripts for extraction of low-level descriptors and application of functionals</li>
</ul>
