# python2.7 -c 'import parallel_dialogues; print parallel_dialogues.par_horn_dialogues("par", 300, 50)'
# python2.7 -c 'import parallel_dialogues; print parallel_dialogues.par_flat_dialogues("par", 300, 50, 2, 1)'
# python2.7 -c 'import parallel_dialogues; print parallel_dialogues.par_flat_dialogues("par", 300, 50, 3, 10)'

import multiprocessing
import conpact2
import cPickle
import numpy as np
import glob

SOFTMAX_WEIGHT = 3
LISTENER_DEPTH = 2
PARTICLE_COUNT = 1000

def simulate_dialogue(r, dialogue, turns):
    for turn in xrange(turns):
        obj, utt, interp, backchannel = dialogue.sample_obj_utt_interp_backchannel(r)
        # Speaker observes listener's interpretation; listener observe
        # speaker's intended meaning
        s_datum = conpact2.SDataInterp(LISTENER_DEPTH, utt, interp)
        l_datum = conpact2.LDataUttWithMeaning(LISTENER_DEPTH - 1,
                                               utt, obj)
        dialogue.new_data([s_datum], [l_datum])

def par_dialogues(outpath, name, f, count, *args):
    p = multiprocessing.Pool()
    jobs = [(i,) + args for i in xrange(count)]
    for result in p.imap_unordered(f, jobs):
        i = result[0]
        print "done: %s %s" % (name, i)
        cPickle.dump(result,
                     open("%s/%s-%s-%03i.pickle"
                          % (outpath, name,
                             "-".join([str(a) for a in args]),
                             i),
                          "w"),
                     protocol=-1)

def horn_dialogue(i_turns):
    i, turns = i_turns
    bgl_domain = conpact2.MemoizedDomain(adjectives=2, objects=2,
                                         max_utterance_length=1,
                                         softmax_weight=SOFTMAX_WEIGHT,
                                         word_cost=[0.5, 1.0],
                                         default_object_prior=[0.8, 0.2])
    r = np.random.RandomState(i)
    flat = np.log([0.5, 0.5])
    def trace_horn_S_lexicalization(turn, dialogue):
        return np.exp(dialogue.uncertain_s_dist(log_object_prior=flat))
    def trace_horn_L_lexicalization(turn, dialogue):
        return np.exp(dialogue.uncertain_l_dist(log_object_prior=flat))

    tracers = {"P-understood": conpact2.trace_P_understood,
               "horn-S-lexicalization": trace_horn_S_lexicalization,
               "horn-L-lexicalization": trace_horn_L_lexicalization,
               }
    dialogue = conpact2.Dialogue(bgl_domain, LISTENER_DEPTH,
                                 i, PARTICLE_COUNT,
                                 tracers=tracers)
    simulate_dialogue(r, dialogue, turns)

    return i, turns, dialogue.traces

def par_horn_dialogues(outpath, count, turns):
    par_dialogues(outpath, "horn", horn_dialogue, count, turns)

def flat_dialogue(i_turns_size_partmult):
    i, turns, size, partmult = i_turns_size_partmult

    domain = conpact2.MemoizedDomain(adjectives=size, objects=size,
                                     max_utterance_length=1,
                                     softmax_weight=SOFTMAX_WEIGHT)
    r = np.random.RandomState(i)
    tracers = {"P-understood": conpact2.trace_P_understood,
               }
    dialogue = conpact2.Dialogue(domain, LISTENER_DEPTH,
                                 i, PARTICLE_COUNT * partmult,
                                 tracers=tracers)
    simulate_dialogue(r, dialogue, turns)

    return i, turns, size, partmult, dialogue.traces

def par_flat_dialogues(outpath, count, turns, size=2, partmult=1):
    par_dialogues(outpath, "flat", flat_dialogue, count, turns, size, partmult)


def extract(pattern, f):
    # I think glob.glob may sort results anyway, and if not it should be
    # stable anyway, but we sort anyway just to be sure in case we want to
    # compare the values from two different calls as parallel arrays.
    for pickle_path in sorted(glob.glob(pattern)):
        result = cPickle.load(open(pickle_path))
        traces = result[-1]
        yield f(traces)

def extract_P_understood(pattern, turn):
    def f(traces):
        return traces["P-understood"][turn]
    return list(extract(pattern, f))

epsilon = 10 ** -2

def gt(x, y):
    return (x - y) > epsilon
def lt(x, y):
    return gt(y, x)

def extract_horn_success(pattern, turn):
    def f(traces):
        S = traces["horn-S-lexicalization"][turn]
        S_success = gt(S[0, 0], S[1, 0]) and lt(S[0, 1], S[1, 1])
        L = traces["horn-L-lexicalization"][turn]
        L_success = gt(L[0, 0], L[0, 1]) and lt(L[1, 0], L[1, 1])
        return S_success and L_success
    return list(extract(pattern, f))

def extract_horn_antisuccess(pattern, turn):
    def f(traces):
        S = traces["horn-S-lexicalization"][turn]
        S_antisuccess = lt(S[0, 0], S[1, 0]) and gt(S[0, 1], S[1, 1])
        L = traces["horn-L-lexicalization"][turn]
        L_antisuccess = lt(L[0, 0], L[0, 1]) and gt(L[1, 0], L[1, 1])
        return S_antisuccess and L_antisuccess
    return list(extract(pattern, f))
