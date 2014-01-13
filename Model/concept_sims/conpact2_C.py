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
import itertools

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
                 adjectives=2, features = 2, max_utterance_length=6,
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
                 default_object_prior=None,
                 # the noise in the w*o representation
                 noise = .1):
        self.adjectives = adjectives
        self.objects = 2**features
        self.features = features
        self.concepts= 3**features
        self.noise = noise
        self.conceptsA = np.array(list(itertools.product([0, 1, 2], repeat=features)))
        self.objectsA = np.array(list(itertools.product([1, 2], repeat=features)))

        # get o_by_c matrix
        self.o_by_c = np.zeros(shape=(self.objects,self.concepts))
        for o in range(0,self.objects):
            current_o = self.objectsA[o,:]
            for c in range(0,self.concepts):
                current_c = self.conceptsA[c,:]
                concept_check = np.zeros(self.features)
                for f in range(0,self.features):
                    if ((current_c[f] == current_o[f]) or (current_c[f] == 0)):
                        concept_check[f] = 1
                    else:
                        concept_check[f] = 0
                if all(concept_check):
                    self.o_by_c[o,c] = 1
                    
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
            lexicon_prior = np.ones((self.adjectives, self.concepts))
        lexicon_prior = np.asarray(lexicon_prior)
        lexicons = []
        for i in xrange(count):
            lexicon = np.empty((self.adjectives, self.concepts))
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
        w_by_o = np.zeros(shape=(self.adjectives,self.objects))

        # generate w_by_o via o_by_c (equivalent to the "lexicon" in the old model)
        for w in range(0, self.adjectives):
            w_by_o[w,:] = self.o_by_c[:,lexicon[w]]

        w_by_o *= np.exp(log_object_prior[np.newaxis, :])


        # add noise
        w_by_o_noise = np.ones(shape=np.shape(w_by_o))
        w_by_o_noise_Z = np.add.reduce(w_by_o_noise, axis=1) # normalize noise
        w_by_o_noise /= w_by_o_noise_Z [:, np.newaxis]
        
        w_by_o_Z = np.add.reduce(w_by_o, axis=1) # normalize w_by_o
        w_by_o /= w_by_o_Z[:, np.newaxis]

        w_by_o_noise *= self.noise 
        w_by_o *= 1-self.noise
        w_by_o += w_by_o_noise
                
        log_wXo= np.log(w_by_o) # put in log space
        
        # literal P(obj|utt) is, by stipulation, proportional to the product
        # of P(obj|word) for all words in utt. Then we multiply by the
        # (context specific) object prior.
        logP_wXo = np.dot(self.utts, log_wXo)
        #logP_wXo += log_object_prior[np.newaxis, :]
        logZ = np.logaddexp.reduce(logP_wXo, axis=1) # get sum (denominator for normalization)
        logP_wXo -= logZ[:, np.newaxis] # normalize
        
        return logP_wXo

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


################################################################
# Simple inference
################################################################

class ExactEnumerationSampler(object):
    # Estimates the posterior on lexicons by keeping a static list of
    # lexicons, and calculating the weight assigned to each.
    
    def __init__(self, domain, lexicon_prior=None, data=[]):
        self._domain = domain
        self._lexicons = np.array(list(itertools.product(range(self._domain.concepts),
                                                         repeat=(self._domain.adjectives))))
        # uniform prior
        if lexicon_prior is None:
            lexicon_prior = np.log(np.ones(len(self._lexicons))/len(self._lexicons))
        self._lexicon_prior = np.asarray(lexicon_prior)
 
        self._weights = lexicon_prior
        
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
