from nips import *

# Galantucci emergence

# XX mention: speaker model of listener being a bit noisy is fine, because
# speaker does an extra layer of pragmatic recursion to clean things up (which
# is why accuracy is so high even when speaker model is noiser than listener
# model)
#
# XX mention: point out that the speaker and listener actually have totally
# different, possibly contradictory data that they're learning from -- but it
# works out in the end

d = dom(adjectives=2, objects=2)
dialogues = simulate_dialogues(4, 10, d)
show_dialogues("emergence2x2-%i.pdf", dialogues)

d = dom(adjectives=3, objects=3)
dialogues = simulate_dialogues(4, 20, d, particle_count=PARTICLES * 10)
show_dialogues("emergence3x3-%i.pdf", dialogues)
