# Mutual exclusivity

# 2 words, 2 objects.
#
# Word 0 is known to refer to object 0. Word 1's meaning is totally unknown.

from nips import *

f = open("ME.log", "w")

d = dom(adjectives=2, objects=2)
lexicon_prior = [[5, 1], [1, 1]]
listener = conpact2.FixedSupportImportanceSampler(d, 0, PARTICLES * 10,
                                                  lexicon_prior=lexicon_prior)
def show_ME_posterior(path, listener):
    show_lex_posterior(path, listener.weighted_lexicons(), (0, 0), (1, 0),
                       xlabel="P(Familiar word literally means familiar object)",
                       ylabel="P(Novel word literally means familiar object)")
show_ME_posterior("ME-prior.pdf", listener)

# Then we hear someone use the novel word. On average we think this word might
# mean anything, so on average it might refer to either, so on average it acts
# like "some" or "glasses" and gets pragmatically strengthened into referring
# to the novel object.

f.write("Novel word means: Familiar object: %0.1f%%. Novel object: %0.1f%%.\n"
        % tuple(100 * np.exp(listener.marginal_dist_n(LISTENER_DEPTH)[1, :])))

# *But* we don't know what was meant. And if it meant the novel object, then
# *that could be because the literal meaning is ambiguous; and if they meant
# *the dog, then that would imply that this novel word does refer to dogs. So
# *on net, we actually end up slightly more convinced that the novel word
# *refers to the familiar object!

show_ME_posterior("ME-1dax.pdf", listener.with_data([conpact2.LDataUtt(LISTENER_DEPTH - 1, 1)]))

# That's for hearing one example. If we hear both words said multiple times in
# this same context, though, then the best explanation is that they're being
# used contrastively, and so we start believing that actually the novel word
# does literally refer to the novel object. Critically, though, it's hearing
# the other, *familiar* word that provides the crucial information here:

data = [conpact2.LDataUtt(LISTENER_DEPTH - 1, 1), conpact2.LDataUtt(LISTENER_DEPTH - 1, 0)] * 10

show_ME_posterior("ME-flat-10dog-10dax.pdf", listener.with_data(data))

# That's for if we have absolutely no prior bias towards or away from
# sparsity. Either is plausible, though -- intuitively if the novel object is
# similar to the familiar object (if the familiar object is a dog, maybe the
# novel object is a cat, or a novel breed of dog), then we might think it's
# more likely that there will be words that refer to both (e.g., "animal"),
# whereas if it's very different (maybe it's a blender or a book), then we
# might think it's unlikely that the same word will refer to both. Both
# possibilities can be articulated in this model:

d = dom(adjectives=2, objects=2)
antisparse_prior = [[5, 1], [2, 2]]
antisparse_listener = conpact2.FixedSupportImportanceSampler(d, 0, PARTICLES * 10, lexicon_prior=antisparse_prior)
sparse_prior = [[5, 1], [0.5, 0.5]]
sparse_listener = conpact2.FixedSupportImportanceSampler(d, 0, PARTICLES * 10, lexicon_prior=sparse_prior)

f.write("To anti-sparse listener, novel word means: "
        "Familiar object: %0.1f%%. Novel object: %0.1f%%.\n"
        % tuple(100 * np.exp(antisparse_listener.marginal_dist_n(LISTENER_DEPTH)[1, :])))
f.write("To sparse listener, novel word means: "
        "Familiar object: %0.1f%%. Novel object: %0.1f%%.\n"
        % tuple(100 * np.exp(sparse_listener.marginal_dist_n(LISTENER_DEPTH)[1, :])))

figure()
show_ME_posterior("ME-antisparse-10dog-10dax.pdf",
                  antisparse_listener.with_data(data))
title("Anti-sparse prior (novel & familiar objects similar)")
figure()
show_ME_posterior("ME-sparse-10dog-10dax.pdf", sparse_listener.with_data(data))
title("Sparse prior (novel & familiar objects dissimilar)")

# (XX: apparently there's a paper of Eve Clark's somewhere where she does something along these lines, Mike will dig it up)
