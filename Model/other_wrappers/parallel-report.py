from nips import *
from parallel_dialogues import *

FLAT2_PATTERN = "par/flat-50-2-*.pickle"
FLAT3_PATTERN = "par/flat-50-3-*.pickle"
HORN_PATTERN = "par/horn-50-*.pickle"

# We have 50, but nothing really interesting happens after 40
TURNS = 40

mean_P_understood = {}
for name, desc, pattern, lty in [("flat2", "2x2 uniform prior", FLAT2_PATTERN,
                                  "k-."),
                                 ("flat3", "3x3 uniform prior", FLAT3_PATTERN,
                                  "r--"),
                                 ("horn", "Horn implicature", HORN_PATTERN,
                                  "g-"),
                      ]:
    mean_P_understood[name] = [np.mean(extract_P_understood(pattern, i))
                               for i in xrange(TURNS)]
    plot(mean_P_understood[name], lty, label=desc, linewidth=2)
ylim(0, 1)
xlim(-0.5, TURNS + 0.5)
legend(loc="lower right")
xlabel(TURN_XLABEL)
ylabel("Mean P(L understands S)")
savefig("emergence-average.pdf", bbox_inches="tight")

mean_horn_success = [np.mean(extract_horn_success(HORN_PATTERN, i))
                     for i in xrange(TURNS)]
mean_horn_antisuccess = [np.mean(extract_horn_antisuccess(HORN_PATTERN, i))
                         for i in xrange(TURNS)]
figure()
plot(mean_horn_success, "g-", linewidth=2, label="Good lexicon")
plot(mean_horn_antisuccess, "b--", linewidth=2, label="Bad lexicon")
ylim(0, 1)
xlim(-0.5, TURNS + 0.5)
xlabel(TURN_XLABEL)
ylabel("Horn lexicalization rate")
legend(loc="upper left")
savefig("emergence-horn-average.pdf", bbox_inches="tight")
