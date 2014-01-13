# Cross-situational learning

# Data available to learning:
#   Two objects are present in each situation, one is always missing
#   Either, one of the two objects is named, by producing its word; or else, the distractor is produced

# Word 0 -> object 0
# Word 1 -> object 1
# Word 2 -> object 2
# and Word 3 is a distractor.
#
# Our learning is invariant to the order of presentation, so there's no point
# in simulating lots of long dialogues... the only thing that matters is the
# relative proportions of different examples, and there the law of large
# numbers will kick in. So we just evaluate the posterior once.
#
# NOTE: this example really pushes the limits of the importance sampling
# (notice that the below graph used 100,000 particles, and still has visible
# asymmetries caused by the sampling; when I tried it with 10,000 particles it
# got the wrong answer! 100,000 particles takes a few minutes on my
# laptop.). This is a perfect place to use MCMC instead.

from nips import *

CS_OBJECTS = 3
CS_WORDS = 4
data_points = []
for missing_obj in xrange(CS_OBJECTS):
    present_objs = range(CS_OBJECTS)
    present_objs.remove(missing_obj)
    object_prior = np.empty(CS_OBJECTS)
    object_prior[missing_obj] = 0.01
    object_prior[present_objs] = (1 - 0.01) / 2
    for target_obj in present_objs:
        data_points.append(conpact2.LDataUtt(LISTENER_DEPTH - 1,
                                             target_obj,
                                             np.log(object_prior)))
    for distractor_word in xrange(CS_OBJECTS, CS_WORDS):
        data_points.append(conpact2.LDataUtt(LISTENER_DEPTH - 1,
                                             distractor_word,
                                             np.log(object_prior)))

d = dom(adjectives=4, objects=3)
learner = conpact2.FixedSupportImportanceSampler(d, 23, PARTICLES * 1000,
                                                 data=data_points * 10)
matshow(np.exp(learner.marginal_dist_n(LISTENER_DEPTH)), cmap="binary",
        vmin=0, vmax=1)
xlabel("Objects")
ylabel("Words")
gcf().savefig("cross-sit.pdf", bbox_inches="tight")
