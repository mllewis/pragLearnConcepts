from nips import *

f = open("horn.log", "w")

# Horn implicature alone

# A trivial version of Leon's common/rare cheap/expensive domain
bgl_domain = dom(adjectives=2, objects=2,
                 word_cost=[0.5, 1.0],
                 default_object_prior=[0.8, 0.2])
dialogue = conpact2.Dialogue(bgl_domain, LISTENER_DEPTH, 0, PARTICLES)

f.write("How speaker refers to \"common\" and \"rare\" objects:\n")
f.write(repr(np.exp(dialogue.uncertain_s_dist())))
f.write("\n\nHow listener interprets \"cheap\" and \"expensive\" words:\n")
f.write(repr(np.exp(dialogue.uncertain_l_dist())))
f.write("\n\n")

DEEP = 20
dialogue_deep = conpact2.Dialogue(bgl_domain, DEEP, 0, PARTICLES * 10)

f.write("How %s-deep speaker refers to \"common\" and \"rare\" objects:\n"
        % (DEEP + 1,))
f.write(repr(np.exp(dialogue_deep.uncertain_s_dist())))
f.write("\n\nHow %s-deep listener interprets \"cheap\" and \"expensive\" words:\n"
        % (DEEP,))
f.write(repr(np.exp(dialogue_deep.uncertain_l_dist())))
f.write("\n\n")

# Emergence + Horn implicature

# XX mention: this is a very general result, can speculate about it as a
# partial cause of things like common words being shorter, rise of useful
# social conventions in general
#
# XX mention: Horn implicature can't survive as implicatures. And connect to
# scalar implicature, which *can* survive.

bgl_dialogues = simulate_dialogues(4, 10, bgl_domain)
show_dialogues("horn-emergence-%i.pdf", bgl_dialogues)

for i, dialogue in enumerate(bgl_dialogues):
    f.write("Dialogue %s, after disabling implicature:\n" % (i,))
    f.write("  How speaker refers to \"common\" and \"rare\" objects:\n")
    f.write(repr(np.exp(dialogue.uncertain_s_dist(log_object_prior=np.log([0.5, 0.5])))))
    f.write("\n  Listener's interpretation of \"cheap\" and \"expensive\" words:\n")
    f.write(repr(np.exp(dialogue.uncertain_l_dist(log_object_prior=np.log([0.5, 0.5])))))
    f.write("\n\n")
