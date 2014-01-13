from nips import *

f = open("scalar-implic.log", "w")

# Demonstrating scalar implicature

some_all_domain = dom(adjectives=2, objects=2)
some_all_prior = [[10, 10], [1, 10]]
dialogue = conpact2.Dialogue(some_all_domain, LISTENER_DEPTH, 0, PARTICLES, lexicon_prior=some_all_prior)

f.write("How speaker refers to \"some\" and \"all\" objects:\n")
f.write(repr(np.exp(dialogue.uncertain_s_dist())))
f.write("\n\nHow listener interprets \"some\" and \"all\" words:\n")
f.write(repr(np.exp(dialogue.uncertain_l_dist())))

def show_scalar_posterior(path, listener):
    show_lex_posterior(path, listener.weighted_lexicons(), (0, 0), (1, 0),
                       xlabel="", ylabel="")

# # Learning scalar implicature

# # two objects: call them "partial x" (xxxooo) and "full x" (xxxxxx)
# #
# # two words: "some" and "all"

# d = dom(adjectives=2, objects=2)
# listener = conpact2.FixedSupportImportanceSampler(d, 0, PARTICLES * 10)

# # Contexts where "partial" and "full" meanings are a priori equiprobable
# some_for_partial_not_full = conpact2.LDataUttWithMeaning(LISTENER_DEPTH - 1, 0, 0)
# all_for_full_not_partial = conpact2.LDataUttWithMeaning(LISTENER_DEPTH - 1, 1, 1)
# # Contexts where either "partial" or "full" is a priori overwhelmingly likely
# some_for_partial_alone = conpact2.LDataUttWithMeaning(LISTENER_DEPTH - 1, 0, 0, log_object_prior=np.log([0.99, 0.01]))

# # Only pragmatically strengthened examples:

# show_scalar_posterior("some-all-only-pragmatic.pdf",
#                       listener.with_data([some_for_partial_not_full,
#                                           all_for_full_not_partial] * 5))

# # A mix of pragmatically strengthened examples, and examples where the meaning
# # from context is obviously 'partial', and 'some' gets used to describe it.

# show_scalar_posterior("some-all-pragmatic+unambiguous.pdf",
#                       listener.with_data([some_for_partial_not_full,
#                                           all_for_full_not_partial,
#                                           some_for_partial_alone] * 5))

# NEW version: set all = ALL, then see what we learn about "some" when we see
# it repeatedly

d = dom(adjectives=2, objects=2)
prior = [[1, 1], [1, 10]]
listener = conpact2.FixedSupportImportanceSampler(d, 0, PARTICLES * 10,
                                                  lexicon_prior=prior)

some_for_partial_not_full = conpact2.LDataUttWithMeaning(LISTENER_DEPTH - 1,
                                                         0, 0)
all_for_full_not_partial = conpact2.LDataUttWithMeaning(LISTENER_DEPTH - 1,
                                                        1, 1)
show_scalar_posterior("some-all-new.pdf",
                      listener.with_data([some_for_partial_not_full,
                                          all_for_full_not_partial] * 5))
