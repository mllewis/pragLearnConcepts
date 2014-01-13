# Second attempt to model conceptual pacts via bayesian pragmatics.

# We imagine a scenario in which a speaker describes an object using a
# sequence of adjectives. The literal speaker S(-1) just samples from a simple
# multinomial lexicon. Specifically, they:
#  - pick an object in the world to describe, according to some prior.
#  - decide how many adjectives to use. This is always observed, so we could
#    put a prior over it but there would be no point.
#  - sample that many times from a multinomial P(word|object). This
#    distribution is called the "lexicon", and is assumed to vary with the
#    context.
# This serves as the base case for the standard probabilistic-game-theoretic
# recursion, with S-1 defined as above, Ln doing rational bayesian inference
# on S(n-1), and Sn sampling utterances from a cost-penalized softmax of
# L(n-1). (NB: this code uses the convention of disjoint indexes for speakers
# and listeners; odd indices are listeners and even indices are speakers.)
#
# We label the above S-1, and make the optimal listener L0, because in
# practice it's a lot easier to reason about words as having meanings than
# about objects as having words. One can write down the literal lexicon in
# either form P(word|obj) or P(obj|word); we choose the latter
# representation. This means that our lexicon actually contains some poorly
# interpreted quantity like "lexical affinity" or "appropriateness" or
# something once we start dealing with multiword utterances, and it's sort of
# arbitrary that we require they sum to one. But whatever; the above provides
# a nice justification, and this works.
#
# So that's our lexicon. Then, we have uncertainty about the lexicon. The
# above determines the joint distribution for utterances and objects for Si/Li
# who *know* what lexicon to use. In fact, the listener is uncertain, so to
# interpret their input, they marginalize over all the ways that some listener
# Ln might have interpreted the input
#    sum_lex Pn(obj|utt, lex) P(lex|data)
# And the speaker is uncertain, so to choose what to say, they marginalize
# over all the ways that some listener might interpret their input
#    sum_lex P(n-1)(obj|utt, lex) P(lex|data)
# and they do a softmax from the resulting marginalized distribution.
#
# Both speaker and listener use a dirichlet prior on each word's meaning, the
# vector lex[word, :]. However, they have different conditioning data. For
# listeners, conditioning data might include:
#   -- that certain utterances were produced (this is informative because it
#      suggests that those utterances are distinctive)
#   -- utterance/meaning pairs
#   -- utterance/non-meaning pairs (if they get correct/incorrect feedback)
# For speakers, conditioning data might include:
#   -- listener guesses at the meaning of some utterances
#   -- whether listener correctly guessed the meaning of some utterances
#      (if all they observe are correct/incorrect without seeing the actual
#      guess)
#   -- whether the listener judged themself to understand (listener says
#      either "ok" or "huh?", and this is somehow related to entropy of
#      listener's posterior on obj|utt).

# Likelihood calculation P_n(utt|obj), P_n(obj|utt), is by enumeration of all
# possible utterances. We use some tricks.
#
# It's *substantially* more efficient to collapse together utterances
# that are equivalent modulo word order (since word order is not
# informative in our model). E.g. for utterances of length 10 on 10
# adjectives, we have:
#   possible utterances including order:   10 ** 10 == 10000000000
#   possible utterances ignoring order: 10 multichoose 10 == 92378
# So we represent utterances as bags-of-words, not lists-of-words.
#
# In general, for each distinct bag-of-words, we must also count how
# many corresponding distinct list-of-words map to it. If the entries
# in each bag were unique (i.e., it were a set), then this would just
# be the number of permutations on k, k!, where k = the number of
# entries. However, if there are duplicate entries in the bag, then
# this will overcount them. For example, here are some of the
# list-of-words that map to the bag {0, 1, 2}:
#   [0, 1, 2]
#   [1, 0, 2]
#   [2, 1, 0]
# But the only list that maps to the bag {0, 0, 0} is
#   [0, 0, 0]
# If item i has r_i duplicates in the multiset, then it produces
# overcounting by a factor of r_i!. So the total multiplicity of each
# multiset is
#   k! / product_i(r_i!)
# (This formula is familiar from the multinomial mass function, for the same
# reason.)
#
# For most purposes, though, we don't actually need to do these calculations!
# They're needed for getting the real S0 matrix. But if our actual goal is to
# get Si/Li for i>0, then we can ignore multiplicities. The idea:
#
# For S0 we compute P(bag_i | obj_j) for all i, j. But we don't use
# the raw numbers, because L1 immediately renormalizes to compute
# P(obj_j | bag_i). This means that all we actually care about are the
# ratios
#   P(bag_i | obj_j) / P(bag_i | obj_k)
# and any transformation which preserves these is fine.
#
# Ignoring the length prior is such a transformation, b/c
#   P(bag_i | obj_j) = P(bag_i | obj_j, len(bag_i)) P(len(bag_i))
# and notice that if we plug this into the above ratio, the
# P(len(bag_i)) terms cancel out. So in fact all we have to compute
# for S0 is
#   P(bag_i | obj_j, len(bag_i))
# and this way we don't have to think about length priors, yay.
#
# Ignoring the multiple relationships between utterance vectors and
# utterance bags is also ok, because:
#   P(bag_i | obj_j)
#     = sum_k P(bag_i | vec_k) P(vec_k | obj_j)
# But P(bag_i | vec_k) is either 0 or 1. And for all the items where
# it is 1, P(vec_k | obj_j) takes the same value. If n_i is the count
# of how many vectors map to bag_i (i.e, sum_k P(bag_i | vec_k)), then
# the above expression is just
#     = n_i P(vec_i | obj_j)
# (in an abuse of notation, we write vec_i to indicate an arbitrary
# representative vector which corresponds to bag_i). And this means
# that
#   P(bag_i | obj_j) / P(bag_i | obj_k)
#     = (n_i P(vec_i | obj_j)) / (n_i P(vec_i | obj_k))
#     = P(vec_i | obj_j) / P(vec_i | obj_k)
#
# So what we actually compute for S0 is the matrix of values
#   P(vec_i | obj_j, len(vec_i))

import sys
from itertools import combinations_with_replacement
from collections import OrderedDict
import numpy as np
from scipy.special import gammaln

################################################################
# The model
################################################################

# This does all of the certain-lexicon likelihood-calculation stuff.
class Domain(object):
    def __init__(self,
                 # Memory usage hint: our probability distribution matrices
                 # will contain this many entries:
                 #    multichoose((adjectives + 1),
                 #                max_utterance_length)
                 #    * objects
                 adjectives=8, objects=10, max_utterance_length=6,
                 # Softmax parameter for the pragmatic speaker:
                 softmax_weight=1,
                 # Word cost for the pragmatic speaker (measured in
                 # nats-per-word); either a single constant or else a vector
                 # of per-word costs:
                 word_cost=1,
                 # The prior on objects to use (may be overridden in any
                 # particular likelihood calculation though). Here this is
                 # specified in linear probability, but all the code below
                 # (e.g., for if you want to override it in some calculation)
                 # specifies it in log probability, always.
                 default_object_prior=None):
        self.adjectives = adjectives
        self.objects = objects
        self.max_utterance_length = max_utterance_length
        self.softmax_weight = softmax_weight
        word_cost = np.asarray(word_cost, dtype=float)
        if word_cost.shape == ():
            word_cost = word_cost * np.ones(self.adjectives)
        self.word_cost = word_cost
        assert word_cost.shape == (self.adjectives,)

        if default_object_prior is None:
            default_object_prior = np.ones(self.objects)
        default_object_prior = np.array(default_object_prior, dtype=float)
        default_object_prior /= np.sum(default_object_prior)
        self.default_log_object_prior = np.log(default_object_prior)

        # We allow utterances of any length up to max_utterance_length. How
        # many distinct such utterances are there? Each arbitrary-length
        # utterance can be thought of as a utterance with exactly
        # max_utterance_length words, where we add an additional "null"
        # word. Except that we disallow the null utterance by fiat, so -1.
        total_utts = -1 + multichoose(self.adjectives + 1,
                                      self.max_utterance_length)
        # Each row of this matrix represents one utterance as a
        # bag-of-words. The bags are represented in count form, e.g. if we
        # have 5 possible adjectives, then the utterance
        #   <"adj0", "adj3", "adj0">
        # is stored as:
        #   [2, 0, 0, 1, 0]
        assert self.max_utterance_length <= 255
        utts = np.zeros((total_utts, self.adjectives), dtype=np.uint8)
        words_with_null = [None] + range(self.adjectives)
        utt_iter = combinations_with_replacement(words_with_null,
                                                 self.max_utterance_length)
        # skip the null utterance:
        skipped = utt_iter.next()
        assert skipped == (None,) * self.max_utterance_length
        # and for all the rest, fill in the corresponding bag:
        for i, utt in enumerate(utt_iter):
            for word in utt:
                if word is not None:
                    utts[i, word] += 1
        assert i + 1 == utts.shape[0]

        self.utts = utts
        self.utt_lengths = np.sum(utts, axis=1)
        self.utt_costs = np.sum(utts * self.word_cost[np.newaxis, :],
                                axis=1)

    def sample_lexicons(self, r, count, lexicon_prior=None):
        if lexicon_prior is None:
            lexicon_prior = np.ones((self.adjectives, self.objects))
        lexicon_prior = np.asarray(lexicon_prior)
        lexicons = []
        for i in xrange(count):
            lexicon = np.empty((self.adjectives, self.objects))
            for j in xrange(self.adjectives):
                lexicon[j, :] = r.dirichlet(lexicon_prior[j, :])
            lexicons.append(lexicon)
        return lexicons

    def log_object_prior(self, log_object_prior=None):
        if log_object_prior is None:
            return self.default_log_object_prior
        else:
            return np.asarray(log_object_prior)

    # These functions all work in log space. They take and return matrices of
    # probabilities shaped like:
    #   (number of utterances, number of objects)
    # each entry of which is either P(obj|utt) or P(utt|obj).
    #
    # I tried implementing it in linear space, and it only went <20% faster
    # end-to-end. (Even when implementing the linear_L0 as np.exp(log_L0),
    # which turns out to be faster since it can exploit np.dot.) I don't think
    # this is worth it given the unknown loss in precision that might bite us
    # down the line; I was hoping for like a 3x speedup or something...

    # The 'lexicon' itself is always written in linear space, though.
    def L0(self, lexicon, log_object_prior=None):
        log_object_prior = self.log_object_prior(log_object_prior)
        log_lexicon = np.log(lexicon)
        # literal P(obj|utt) is, by stipulation, proportional to the product
        # of P(obj|word) for all words in utt. Then we multiply by the
        # (context specific) object prior.
        logP_L = np.dot(self.utts, log_lexicon)
        logP_L += log_object_prior[np.newaxis, :]
        logZ = np.logaddexp.reduce(logP_L, axis=1)
        logP_L -= logZ[:, np.newaxis]
        return logP_L

    def S_minus_one(self, lexicon):
        # 'lexicon' is an L0 lexicon. Figure out the corresponding P(word|obj)
        # for a speaker that would lead to this effective lexicon if L_0 were
        # doing rational inference over S_{-1}.
        # This sub-lexicon is an unknown matrix, that satisfied two
        # constraints:
        # - each column sums to 1
        # - each entry, divided by its row's sum, equals the corresponding
        #   entry in the lexicon
        #   - which means, that lex entry times the row's sum equals the
        #     sublex entry, so this is a linear constraint.
        # We unravel sublex into a vector and solve for it.
        # There are 'objects' constraints of the first kind, and 'objects x
        # adjectives' constraints of the second kind. (Yes, this means some of
        # them are redundant. Whatever.)
        A = np.empty((self.objects + (self.adjectives * self.objects),
                      self.adjectives * self.objects))
        b = np.empty(self.objects + (self.adjectives * self.objects))
        constraint_i = 0
        for j in xrange(self.objects):
            # Each column sums to 1
            constraint = np.zeros((self.adjectives, self.objects))
            constraint[:, j] = 1
            A[constraint_i, :] = constraint.ravel()
            b[constraint_i] = 1
            constraint_i += 1
        for i in xrange(self.adjectives):
            for j in xrange(self.objects):
                constraint = np.zeros((self.adjectives, self.objects))
                # sublex[i, j] = lexicon[i, j] * sum(sublex[i, :])
                # sublex[i, j] - lexicon[i, j] * sum(sublex[i, :]) = 0
                constraint[i, j] = 1
                constraint[i, :] -= lexicon[i, j]
                A[constraint_i, :] = constraint.ravel()
                b[constraint_i] = 0
                constraint_i += 1
        sublex = np.dot(np.linalg.pinv(A), b).reshape(lexicon.shape)
        import pdb; pdb.set_trace()
        return np.log(sublex)

    def L(self, logP_S, log_object_prior=None):
        # logP_S is P(utt|obj, ...) as an (utt, object) matrix
        # to convert to P(obj|utt, ...), we scale each column by the prior
        # P(object), and then normalize each row.
        log_object_prior = self.log_object_prior(log_object_prior)
        logP_L = logP_S + log_object_prior[np.newaxis, :]
        logZ = np.logaddexp.reduce(logP_L, axis=1)
        logP_L -= logZ[:, np.newaxis]
        return logP_L

    def S(self, logP_L):
        # logP_L is P(object|utt, ...) as an (utt, object) matrix
        # For the speaker:
        #   P(utt|object) = 1/Z * exp(SOFTMAX_WEIGHT * U)
        # where
        #   U = log P_L(object | utt) - cost(utt)
        U = logP_L - self.utt_costs[:, np.newaxis]
        U *= self.softmax_weight
        logZ = np.logaddexp.reduce(U, axis=0)
        U -= logZ[np.newaxis, :]
        return U

    def dist_n(self, n, lexicon, log_object_prior=None):
        assert n >= 0
        log_object_prior = self.log_object_prior(log_object_prior)
        if n % 2 == 0:
            # Listener
            if n == 0:
                return self.L0(lexicon, log_object_prior)
            else:
                return self.L(self.dist_n(n - 1, lexicon, log_object_prior),
                              log_object_prior)
        else:
            assert n % 2 == 1 # Speaker
            return self.S(self.dist_n(n - 1, lexicon, log_object_prior))

def test_Domain_basics():
    d = Domain(adjectives=2, objects=4, max_utterance_length=3,
               default_object_prior=[1, 2, 3, 4])
    assert d.adjectives == 2
    assert d.objects == 4
    assert d.max_utterance_length == 3
    assert d.utts.shape == (9, 2)
    assert np.array_equal(d.utts,
                          [[1, 0],
                           [0, 1],
                           [2, 0],
                           [1, 1],
                           [0, 2],
                           [3, 0],
                           [2, 1],
                           [1, 2],
                           [0, 3]])
    assert np.array_equal(d.utt_lengths, [1, 1, 2, 2, 2, 3, 3, 3, 3])
    assert np.allclose(d.default_log_object_prior,
                       np.log([0.1, 0.2, 0.3, 0.4]))

def test_Domain_L0():
    d = Domain(adjectives=2, objects=2, max_utterance_length=2)

    lexicon = np.asarray([[0.1, 0.9],
                          [0.2, 0.8]])
    L0 = d.L0(lexicon, np.log([0.3, 0.7]))
    L0_raw = np.exp(L0)
    # 5 possible utterances
    expected_raw = np.array([
        # utterance: <0>
        [0.1 * 0.3, 0.9 * 0.7],
        # utterance: <1>
        [0.2 * 0.3, 0.8 * 0.7],
        # utterance: <0 0>
        [0.1 ** 2 * 0.3, 0.9 ** 2 * 0.7],
        # utterance: <0 1>
        [0.1 * 0.2 * 0.3, 0.9 * 0.8 * 0.7],
        # utterance: <1 1>
        [0.2 ** 2 * 0.3, 0.8 ** 2 * 0.7],
        ])
    for i in xrange(5):
        expected_raw[i, :] /= np.sum(expected_raw[i, :])
    assert np.allclose(L0_raw, expected_raw)

def test_Domain_L():
    d = Domain(adjectives=2, objects=2, max_utterance_length=2)
    before_L = np.array([[0.4, 0.2],
                         [0.6, 0.8],
                         [0.1, 0.7],
                         [0.4, 0.1],
                         [0.5, 0.2]])
    # Uniform prior
    after_L_expected = np.array([[0.4 / 0.6, 0.2 / 0.6],
                                 [0.6/ 1.4, 0.8 / 1.4],
                                 [0.1 / 0.8, 0.7 / 0.8],
                                 [0.4 / 0.5, 0.1 / 0.5],
                                 [0.5 / 0.7, 0.2 / 0.7]])
    assert np.allclose(np.sum(after_L_expected, axis=1), 1)
    assert np.allclose(np.exp(d.L(np.log(before_L), np.asarray([0, 0]))),
                       after_L_expected)
    # Non-uniform prior
    prior = np.log([0.4, 0.6])
    after_L_expected_nonunif = np.array([[0.4 * 0.4, 0.2 * 0.6],
                                         [0.6 * 0.4, 0.8 * 0.6],
                                         [0.1 * 0.4, 0.7 * 0.6],
                                         [0.4 * 0.4, 0.1 * 0.6],
                                         [0.5 * 0.4, 0.2 * 0.6]])
    after_L_expected_nonunif /= np.sum(after_L_expected_nonunif,
                                       axis=1)[:, np.newaxis]
    assert np.allclose(np.exp(d.L(np.log(before_L), prior)),
                       after_L_expected_nonunif)

def test_Domain_S():
    d = Domain(adjectives=2, objects=2, max_utterance_length=2,
              softmax_weight=7, word_cost=[2, 3])
    P_L = np.array([[0.1, 0.9],  # utt <0>
                    [0.2, 0.8],  # utt <0>
                    [0.3, 0.7],  # utt <0 0>
                    [0.4, 0.6],  # utt <0 1>
                    [0.5, 0.5]]) # utt <1 1>
    P_S_unnorm = np.array([[0.1 ** 7 / np.exp(7 * 2),
                            0.9 ** 7 / np.exp(7 * 2)],
                           [0.2 ** 7 / np.exp(7 * 3),
                            0.8 ** 7 / np.exp(7 * 3)],
                           [0.3 ** 7 / np.exp(7 * (2 + 2)),
                            0.7 ** 7 / np.exp(7 * (2 + 2))],
                           [0.4 ** 7 / np.exp(7 * (2 + 3)),
                            0.6 ** 7 / np.exp(7 * (2 + 3))],
                           [0.5 ** 7 / np.exp(7 * (3 + 3)),
                            0.5 ** 7 / np.exp(7 * (3 + 3))],
                           ])
    P_S = P_S_unnorm / np.sum(P_S_unnorm, axis=0)
    assert np.allclose(np.sum(P_S[:, 0]), 1)
    assert np.allclose(np.sum(P_S[:, 1]), 1)

    assert np.allclose(np.exp(d.S(np.log(P_L))), P_S)

def test_Domain_iterates():
    d = Domain(adjectives=2, objects=3, max_utterance_length=4,
              softmax_weight=3, word_cost=2)
    r = np.random.RandomState(0)
    lexicon = r.rand(2, 3)
    object_prior = [0.2, 0.5, 0.3]
    L0 = d.L0(lexicon, np.log(object_prior))
    S1 = d.S(L0)
    L2 = d.L(S1, np.log(object_prior))
    S3 = d.S(L2)
    assert np.allclose(d.dist_n(0, lexicon, object_prior), L0)
    assert np.allclose(d.dist_n(1, lexicon, object_prior), S1)
    assert np.allclose(d.dist_n(2, lexicon, object_prior), L2)
    assert np.allclose(d.dist_n(3, lexicon, object_prior), S3)

################################################################
# "Data points" available to a speaker/listener, used to condition posterior
# over lexicon
################################################################

# XX a number of types of listener data could be consolidated as a single
# observation that (1) some utterance was produced by a given speaker using a
# particular lexicon and prior, (2) we have some distribution over what the
# speaker might have been saying, which *may be different* from the prior they
# used. E.g. if we have learned the true meaning, or have ruled out some
# meaning (because we guessed but then were told we were wrong). Only
# trickiness is that some of these involve exact zeros in the distribution, ->
# -infs in the log probability.

# Listener data
class LDataUtt(object):
    # Just an utterance that was made, meaning unknown. But we can still learn
    # something from the fact that the speaker thought it was informative...
    def __init__(self, speaker, utt, log_object_prior=None):
        assert speaker % 2 == 1
        self._speaker = speaker
        self._log_object_prior = log_object_prior
        self._utt = utt

    def log_likelihood(self, domain, lexicon):
        log_object_prior = domain.log_object_prior(self._log_object_prior)
        logP_S = domain.dist_n(self._speaker, lexicon, log_object_prior)
        return np.logaddexp.reduce(logP_S[self._utt, :] + log_object_prior)

# Listener data
class LDataUttWithMeaning(object):
    # An utterance that a speaker produced, and whose underlying meaning was
    # somehow observed.
    def __init__(self, speaker, utt, obj, log_object_prior=None):
        assert speaker % 2 == 1
        self._speaker = speaker
        self._utt = utt
        self._obj = obj
        self._log_object_prior = log_object_prior

    def log_likelihood(self, domain, lexicon):
        logP_S = domain.dist_n(self._speaker, lexicon, self._log_object_prior)
        return logP_S[self._utt, self._obj]

# Speaker data
class SDataInterp(object):
    # We know how a listener interpreted some utterance
    def __init__(self, listener, utt, obj, log_object_prior=None):
        assert listener % 2 == 0
        self._listener = listener
        self._utt = utt
        self._obj = obj
        self._log_object_prior = log_object_prior

    def log_likelihood(self, domain, lexicon):
        logP_L = domain.dist_n(self._listener, lexicon,
                               self._log_object_prior)
        return logP_L[self._utt, self._obj]

# Speaker data
class SDataBackchannel(object):
    # The listener sampled an interpretation of an utterance, and then
    # calculated the probability of their interpretation being correct, and
    # then said "ok" with that probability and "huh?" with 1-that
    # probability. (XX maybe we should use the domain's softmax parameter
    # here? This is effectively a softmax=1 strategy.) So we have
    #   P(ok|utt) = sum_obj P(ok|utt, obj) P(obj|utt)
    #             = sum_obj P(obj|utt) P(obj|utt)
    # So this is sort of like getting an estimate of the listener's
    # entropy -- except instead of expected log probability, it's expected
    # probability, and then we see a sample instead of the actual number.
    def __init__(self, listener, utt, feedback, log_object_prior=None):
        # 'feedback' is a string, either "ok" or "huh?"
        assert listener % 2 == 0
        self._listener = listener
        self._utt = utt
        if feedback == "ok":
            self._maybe_understood = True
        elif feedback == "huh?":
            self._maybe_understood = False
        else:
            assert False
        self._log_object_prior = log_object_prior

    def log_likelihood(self, domain, lexicon):
        logP_L = domain.dist_n(self._listener, lexicon,
                               self._log_object_prior)
        P_ok = np.sum(np.exp(2 * logP_L[self._utt, :]))
        if self._maybe_understood:
            return np.log(P_ok)
        else:
            return np.log(1 - P_ok)

# XX not used
def prior_lexicon_loglik_unnorm(lexicon, lexicon_prior):
    # lexicon_prior is a matrix of dirichlet parameters
    # each column of 'lexicon_prior' corresponds to each column of 'lexicon'.
    # We ignore the normalization on the dirichlet distribution, because it
    # depends only on 'lexicon_prior', not 'lexicon'.
    return np.sum((lexicon_prior - 1) * np.log(lexicon))

# XX not used
def logP_lexicon_given_data_unnorm(lexicon, lexicon_prior, data):
    # P(lexicon | data) = P(data | lexicon) P(lexicon) / Z
    total = prior_lexicon_loglik_unnorm(lexicon_prior, lexicon)
    for datum in data:
        total += datum.log_likelihood(lexicon)
    return total

################################################################
# Memoizing version of the core likelihood code (in memory for now, persistent
# maybe later if it turns out to be justified)
################################################################

class MemoizedDomain(Domain):
    # This is great for the fixed support importance "sampler", because it
    # calculates the same distributions for the same lexicons over and over
    # again, so unconditionally caching them is a big win (and has bounded
    # memory overhead). For something like an MH- or slice-estimator, it would
    # be much less useful and also leak memory.
    def __init__(self, *args, **kwargs):
        Domain.__init__(self, *args, **kwargs)
        self._cache = {}

    def dist_n(self, n, lexicon, log_object_prior=None):
        log_object_prior = self.log_object_prior(log_object_prior)
        key = (n, array_to_key(lexicon), array_to_key(log_object_prior))
        if key not in self._cache:
            self._cache[key] = Domain.dist_n(self, n, lexicon,
                                             log_object_prior)
        return self._cache[key]

################################################################
# Simple inference
################################################################

class EmceeSampler(object):
    # To parametrize the simplices that our lexica live on, we introduce
    # hypothetical auxiliary variables T_i. (T is for Total). The scheme is,
    # we interpret a vector of unconstrained parameters theta_j as
    #   T = sum_j theta_j
    #   p_j = theta_j / T
    # and then the p_j parameters are the points on the lexicon simplex (for
    # one row of the lexicon), and T is allowd to vary arbitrarily but with a
    # simple diffuse prior to make sure the sampling is all hunky-dory.
    def __init__(self, domain, lexicon_prior=None, data=[]):
        pass

class FixedSupportImportanceSampler(object):
    # Estimates the posterior on lexicons by keeping a static list of
    # lexicons, and calculating the weight assigned to each.
    # This is stateful -- you can add new data and it updates its estimates.
    # Basically this is a trivial particle filter with no resampling. The
    # advantage is that it's very fast to update and sample from even as the
    # available data shifts (because we do the pragmatic inference once for
    # each particle and then can re-use that work when conditioning on new
    # data). The disadvantage is the standard disadvantage of importance
    # sampling -- it suffers from the curse of dimensionality. So this scales
    # to long utterances and long dialogues, but doesn't scale to large
    # lexicons.
    def __init__(self, domain, particle_seed, particle_count,
                 lexicon_prior=None, data=[]):
        self._domain = domain
        if lexicon_prior is None:
            lexicon_prior = np.ones((self._domain.adjectives,
                                     self._domain.objects))
        self._lexicon_prior = np.asarray(lexicon_prior)
        # We use the prior on lexicons to sample the particle values for our
        # importance sampler. There's a subtlety here; the proper way to do
        # this is to set each particle's weight to P_prior(lexicon) /
        # P_proposal(lexicon) and then update it further from there as next
        # data comes in. But, because we use the prior to choose our initial
        # distribution, the two terms in this ratio cancel out, so we just
        # initialize our weights to be all-zero.
        r = np.random.RandomState(particle_seed)
        self._lexicons = self._domain.sample_lexicons(r, particle_count,
                                                      lexicon_prior)
        self._weights = np.zeros(len(self._lexicons))
        self._data = []
        if data:
            self.add_data(data)

    def add_data(self, data):
        self._data += data
        for i, lexicon in enumerate(self._lexicons):
            for datum in data:
                self._weights[i] += datum.log_likelihood(self._domain, lexicon)

    # Returns a *copy* of this sampler, on which .add_data has been called,
    # leaving the original sampler unchanged.
    # Not so efficient, but helpful for interactive use.
    def with_data(self, data):
        import copy
        # Shallow copy
        new_obj = copy.copy(self)
        # And copy the mutable attributes
        new_obj._data = copy.copy(new_obj._data)
        new_obj._weights = np.copy(new_obj._weights)
        new_obj.add_data(data)
        return new_obj

    def weighted_lexicons(self):
        # Returns
        Z = np.logaddexp.reduce(self._weights)
        self._weights -= Z
        return self._weights, self._lexicons

    def marginal_dist_n(self, n, log_object_prior=None):
        log_object_prior = self._domain.log_object_prior(log_object_prior)
        marginal_dist = None
        for logP, lexicon in zip(*self.weighted_lexicons()):
            dist = self._domain.dist_n(n, lexicon, log_object_prior)
            if marginal_dist is None:
                marginal_dist = np.empty(dist.shape)
                marginal_dist[:] = -np.inf
            np.logaddexp(marginal_dist, logP + dist, out=marginal_dist)
        return marginal_dist

class Dialogue(object):
    def __init__(self, domain, listener, particle_seed, particle_count,
                 tracers=None,
                 lexicon_prior=None,
                 speaker_lexicon_prior=None,
                 listener_lexicon_prior=None):
        assert listener % 2 == 0
        self.domain = domain
        self.listener = listener
        if speaker_lexicon_prior is None:
            speaker_lexicon_prior = lexicon_prior
        self.speaker_sampler = FixedSupportImportanceSampler(self.domain,
                                                             particle_seed,
                                                             particle_count,
                                                             lexicon_prior=speaker_lexicon_prior)
        if listener_lexicon_prior is None:
            listener_lexicon_prior = lexicon_prior
        listener_p_seed = 12345 + particle_seed
        self.listener_sampler = FixedSupportImportanceSampler(self.domain,
                                                              listener_p_seed,
                                                              particle_count,
                                                              lexicon_prior=listener_lexicon_prior)
        if tracers is None:
            tracers = default_tracers()
        self.tracers = tracers
        self.traces = {}
        for name, tracer in self.tracers.iteritems():
            self.traces[name] = OrderedDict()
        self.turn = 0
        self._trace()

    def uncertain_l_dist(self, log_object_prior=None):
        s = self.listener_sampler
        # return s.marginal_dist_n(self.listener, log_object_prior)
        subjective_s_dist = s.marginal_dist_n(self.listener - 1, log_object_prior)
        return self.domain.L(subjective_s_dist)

    def uncertain_s_dist(self, log_object_prior=None):
        s = self.speaker_sampler
        subjective_l_dist = s.marginal_dist_n(self.listener, log_object_prior)
        return self.domain.S(subjective_l_dist)

    def new_data(self, speaker_data, listener_data):
        self.speaker_sampler.add_data(speaker_data)
        self.listener_sampler.add_data(listener_data)
        self.turn += 1
        self._trace()

    def _trace(self):
        for name, tracer in self.tracers.iteritems():
            value = tracer(self.turn, self)
            if value is not None:
                self.traces[name][self.turn] = value

    def sample_obj(self, r, log_object_prior=None):
        log_object_prior = self.domain.log_object_prior(log_object_prior)
        return weighted_choice(r, np.exp(log_object_prior))

    def sample_utt(self, r, obj, log_object_prior=None):
        s_dist = self.uncertain_s_dist(log_object_prior)
        return weighted_choice(r, np.exp(s_dist[:, obj]))

    def sample_interp_backchannel(self, r, utt, log_object_prior=None):
        l_dist = self.uncertain_l_dist(log_object_prior)
        interp = weighted_choice(r, np.exp(l_dist[utt, :]))
        if r.rand() < np.exp(l_dist[utt, interp]):
            return interp, "ok"
        else:
            return interp, "huh?"

    def sample_obj_utt_interp_backchannel(self, r, log_object_prior=None):
        obj = self.sample_obj(r, log_object_prior)
        utt = self.sample_utt(r, obj, log_object_prior)
        interp, backchannel = self.sample_interp_backchannel(r, utt,
                                                             log_object_prior)
        return (obj, utt, interp, backchannel)

# Tracer objects

def with_args(trace_fn, *extra_args, **extra_kwargs):
    def wrapped_trace_fn(*args):
        return trace_fn(*(args + extra_args), **extra_kwargs)
    return wrapped_trace_fn

def trace_symKL(turn, dialogue, log_object_prior=None):
    # Calculate the marginal listener distribution as estimated by the
    # listener, and as estimated by the speaker, and check to what extent
    # these agree with each other in KL terms.
    dists = []
    for s in (dialogue.speaker_sampler, dialogue.listener_sampler):
        dist = s.marginal_dist_n(dialogue.listener, log_object_prior)
        dists.append(dist)
    return (np.sum((dists[0] - dists[1]) * np.exp(dists[0]))
            + np.sum((dists[1] - dists[0]) * np.exp(dists[1])))

def with_freq(n, trace_fn):
    def wrapped_trace_fn(turn, dialogue):
        if turn % n != 0:
            return None
        else:
            return trace_fn(turn, dialogue)
    return wrapped_trace_fn

def trace_speaker_lexicon_posterior(turn, dialogue):
    weights, lexicon = dialogue.speaker_sampler.weighted_lexicons()
    return weights.copy(), lexicon

def trace_listener_lexicon_posterior(turn, dialogue):
    weights, lexicon = dialogue.listener_sampler.weighted_lexicons()
    return weights.copy(), lexicon

def trace_P_understood(turn, dialogue, log_object_prior=None):
    # P(correct_understanding)
    #    = sum_obj,utt P_S(utt|obj) P_L(obj|utt) P(obj)
    log_object_prior = dialogue.domain.log_object_prior(log_object_prior)
    s_dist = dialogue.uncertain_s_dist(log_object_prior)
    l_dist = dialogue.uncertain_l_dist(log_object_prior)
    return np.sum(np.exp(s_dist
                         + l_dist
                         + log_object_prior[np.newaxis, :]))

# Traces the meaning of single-word utterances according to the speaker's
# model of the listener, and the listener's model of the listener.
def trace_speaker_single_word_meanings(turn, dialogue,
                                       log_object_prior=None):
    dist = dialogue.speaker_sampler.marginal_dist_n(dialogue.listener,
                                                    log_object_prior)
    return np.exp(dist[:dialogue.domain.adjectives, :])

def trace_listener_single_word_meanings(turn, dialogue,
                                        log_object_prior=None):
    dist = dialogue.listener_sampler.marginal_dist_n(dialogue.listener,
                                                     log_object_prior)
    return np.exp(dist[:dialogue.domain.adjectives, :])

def trace_expected_length(turn, dialogue, log_object_prior=None):
    s_dist = dialogue.uncertain_s_dist(log_object_prior)
    log_object_prior = dialogue.domain.log_object_prior(log_object_prior)
    # compute P(utt)
    linear_utt_dist = np.sum(np.exp(s_dist + log_object_prior[np.newaxis, :]),
                             axis=1)
    assert np.allclose(np.sum(linear_utt_dist), 1)
    return np.sum(linear_utt_dist * dialogue.domain.utt_lengths)

def trace_length_dist(turn, dialogue, log_object_prior=None):
    s_dist = dialogue.uncertain_s_dist(log_object_prior)
    log_object_prior = dialogue.domain.log_object_prior(log_object_prior)
    # compute P(utt)
    linear_utt_dist = np.sum(np.exp(s_dist + log_object_prior[np.newaxis, :]),
                             axis=1)
    # Sum the probability mass over all same-length utterances
    import pandas
    length_dist = (pandas.Series(linear_utt_dist)
                     .groupby(dialogue.domain.utt_lengths)
                       .aggregate(np.sum))
    return length_dist

def default_tracers(posterior_freq=5):
    return {
        "symKL": trace_symKL,
        "speaker-lexicon-posterior":
            with_freq(posterior_freq, trace_speaker_lexicon_posterior),
        "listener-lexicon-posterior":
            with_freq(posterior_freq, trace_listener_lexicon_posterior),
        "P-understood": trace_P_understood,
        "speaker-single-word-meanings": trace_speaker_single_word_meanings,
        "listener-single-word-meanings": trace_listener_single_word_meanings,
        "expected-length": trace_expected_length,
        "length-dist": trace_length_dist,
    }

################################################################
# Basic probability stuff
################################################################

# XX not used
# https://en.wikipedia.org/wiki/Dirichlet_distribution#Probability_density_function
def log_multinomial_beta(alpha):
    return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))

# XX not used
def log_dirichlet_likelihood(alpha, x):
    return np.sum((alpha - 1) * np.log(x)) - log_multinomial_beta(alpha)

################################################################
# numpy.random.choice
################################################################

# A similar function is available in numpy 1.7, but I only have 1.6 (and
# ipython is somehow making it annoying to upgrade, so whatever)

def weighted_choice(r, weights):
    cdf = np.cumsum(weights, dtype=float)
    quantile = r.uniform(low=0, high=cdf[-1])
    return np.searchsorted(cdf, quantile)

def test_weighted_choice():
    r = np.random.RandomState(0)
    counts = [0, 0, 0]
    for i in xrange(1000):
        counts[weighted_choice(r, [0.1, 0.5, 0.4])] += 1
    assert np.sum(counts) == 1000
    assert (0.1 * 1000) * 0.9 < counts[0] < (0.1 * 1000) * 1.1
    assert (0.5 * 1000) * 0.9 < counts[1] < (0.5 * 1000) * 1.1
    assert (0.4 * 1000) * 0.9 < counts[2] < (0.4 * 1000) * 1.1

################################################################
# Basic combinatoric functions
################################################################

# https://en.wikipedia.org/wiki/Falling_factorial_power
# n * (n - 1) * ... * (n - (k - 1))
def falling_factorial(n, k):
    assert 0 <= k <= n
    total = 1
    for i in xrange(k):
        total *= (n - i)
    return total

def test_falling_factorial():
    assert falling_factorial(4, 0) == 1
    assert falling_factorial(4, 1) == 4
    assert falling_factorial(4, 2) == (4 * 3)
    assert falling_factorial(4, 3) == (4 * 3 * 2)
    assert falling_factorial(4, 4) == (4 * 3 * 2 * 1)

def factorial(n):
    return falling_factorial(n, n)

def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(2) == 2
    assert factorial(3) == 6
    assert factorial(4) == 24

def choose(n, k):
    return falling_factorial(n, k) // factorial(k)

def test_choose():
    assert choose(0, 0) == 1
    for n in xrange(5):
        for k in xrange(n):
            # the Pascal's triangle recurrence
            assert choose(n + 1, k + 1) == choose(n, k) + choose(n, k + 1)

def multichoose(n, k):
    return choose(n + k - 1, k)

def test_multichoose():
    for n in xrange(4):
        for k in xrange(2 * n):
            multisets = list(combinations_with_replacement(xrange(n), k))
            assert len(multisets) == multichoose(n, k)

################################################################
# A shared cache for the big likelihood matrices
################################################################

DEFAULT_CACHE_SIZE = 3000 # megabytes

# XX not used
class MemLimitedLRU(object):
    """LRU cache (key -> numpy array), limited by the total size of the value
    arrays."""
    def __init__(self, megabytes_limit):
        self._bytes_size = 0
        self._cache = OrderedDict()
        self.set_limit(megabytes_limit)

    def set_limit(self, megabytes):
        self._bytes_limit = megabytes * (2 ** 20)
        self._flush()

    def try_get(self, key):
        if key in self._cache:
            value = self._cache[key]
            # Push it to the end
            del self._cache[key]
            self._cache[key] = value
            return value
        else:
            return None

    def _arr_size(self, arr):
        return arr.size * arr.dtype.itemsize

    def store(self, key, value):
        assert key not in self._cache
        assert isinstance(value, np.ndarray)
        self._cache[key] = value
        self._bytes_size += self._arr_size(value)
        self._flush()

    def _flush(self):
        while self._bytes_size > self._bytes_limit:
            evicted_key, evicted_value = self._cache.popitem(last=False)
            self._bytes_size -= self._arr_size(evicted_value)

_global_cache = MemLimitedLRU(DEFAULT_CACHE_SIZE)
cache_try_get = _global_cache.try_get
cache_store = _global_cache.store
cache_set_limit = _global_cache.set_limit

def test_MemLimitedLRU():
    # Holds 1 MiB, a400 is 400 KiB, so 2 cache will hold 2 copies but not 3
    lru = MemLimitedLRU(1)
    a400 = np.ones(200 * (2 ** 10), dtype=np.uint16)

    lru.store(1, a400)
    lru.store(2, a400)
    lru.store(3, a400)
    assert lru.try_get(1) is None
    assert lru.try_get(2) is a400
    # Now 3 is LRU, even though it was added last
    lru.store(4, a400)
    assert lru.try_get(3) is None
    assert lru.try_get(2) is a400
    assert lru.try_get(4) is a400
    # This should cause 2 to be evicted as LRU
    lru.set_limit(0.5)
    assert lru.try_get(2) is None
    assert lru.try_get(4) is a400

################################################################
# Utility to use arrays as dict keys, useful for caching
################################################################

# Strategy: since this is intended for caching, we error on the side of false
# negatives rather than false positives. So two arrays might compare
# identical with ==, but be considered different by this function. E.g. these
# will be considered different:
#   [1, 2] versus [1.0, 2.0]
#   same array with different endianness
# However, we don't take array storage order (C/F/discontiguous/etc.) into
# account. (Anything except C-contiguous will be a little more expensive
# though.) Basically we compare dtype, shape, and raw memory contents.
import hashlib
def array_to_key(arr):
    arr = np.asarray(arr)
    if not arr.flags.c_contiguous:
        arr = arr.copy(order="C")
    data_hash = hashlib.sha1(arr).digest()
    return (arr.dtype, arr.shape, data_hash)

def test_array_to_key():
    a = np.array([1, 2, 3, 4], dtype=np.int64)
    a_f = np.asarray(a, dtype=np.float64)
    a_cast = a.view(np.float64)
    a_reshaped = a.reshape((2, 2))
    arrs = [a, a_f, a_cast, a_reshaped]
    for arr in arrs:
        assert array_to_key(arr) == array_to_key(arr)
        assert array_to_key(arr) == array_to_key(arr.copy())
        for arr2 in arrs:
            if arr is not arr2:
                assert array_to_key(arr) != array_to_key(arr2)

    a_discont = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)[::2]
    assert array_to_key(a_discont) == array_to_key(a)

################################################################
# If this file is run as a script, run tests
################################################################

if __name__ == "__main__":
    import nose
    nose.runmodule()
