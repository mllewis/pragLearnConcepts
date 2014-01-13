# -*- coding: utf-8 -*-
import conpact2_C
import numpy as np
# We need ~this as a weight for the Horn implicature emergence to work, so let's just use it everywhere
SOFTMAX_WEIGHT = 3
# Resolution of the importance sampler
PARTICLES = 1000
# Listener is L2-with-uncertainty, speaker is S3-with-uncertainty
LISTENER_DEPTH = 2
CONVERGENCE_FIG_WIDTH = 20

TURN_XLABEL = "Dialogue turn"

import matplotlib
#matplotlib.use("Cairo")
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
matplotlib.rcParams["font.size"] = 24
for tick in ["xtick", "ytick"]:
    for size in ["major", "minor"]:
        for attr in ["pad", "size"]:
            matplotlib.rcParams["%s.%s.%s" % (tick, size, attr)] *= 2
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

# Use a fixed max_utterance_length and softmax weight for everything
# With length-1 utterances we could get away with using actual sampling for posteriors...
def dom(**kwargs):
    return conpact2_C.MemoizedDomain(max_utterance_length=1, softmax_weight=SOFTMAX_WEIGHT, **kwargs)

def show_lex_posterior(path, weights_particles, idx1, idx2,
                       labels=True, xlabel=None, ylabel=None):
    weights, particles = weights_particles
    masses, x_edges, y_edges = np.histogram2d([p[idx1] for p in particles],
                                              [p[idx2] for p in particles],
                                              range=[[0, 1], [0, 1]],
                                              weights=np.exp(weights))
    f = figure()
    imshow(masses.T, aspect="equal",
           extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
           origin="lower", vmin=0, cmap="binary", interpolation="nearest")
    if labels:
        if xlabel is None:
            plt.xlabel("P(word %s means obj %s | data)" % idx1)
        else:
            plt.xlabel(xlabel)
        if ylabel is None:
            plt.ylabel("P(word %s means obj %s | data)" % idx2)
        else:
            plt.ylabel(ylabel)
    savefig(path, bbox_inches="tight")
    close(f)

def simulate_dialogues(count, turns_per_dialogue, domain, lexicon_prior=None,
                       listener_depth=LISTENER_DEPTH,
                       particle_seed_base=0, particle_count=PARTICLES):
    listener_speaker_depth = listener_depth - 1
    dialogues = []
    for i in xrange(count):
        r = np.random.RandomState(i)
        dialogue = conpact2_C.Dialogue(domain, listener_depth,
                                     particle_seed_base + i,
                                     particle_count,
                                     lexicon_prior=lexicon_prior)
        dialogues.append(dialogue)
        for turn in xrange(turns_per_dialogue):
            obj, utt, interp, backchannel = dialogue.sample_obj_utt_interp_backchannel(r)
            # Speaker observes listener's interpretation; listener observe speaker's intended meaning
            s_datum = conpact2_C.SDataInterp(listener_depth, utt, interp)
            l_datum = conpact2_C.LDataUttWithMeaning(listener_speaker_depth,
                                                   utt, obj)
            dialogue.new_data([s_datum], [l_datum])
    return dialogues

def show_dialogue(path, dialogue):
    fig = figure(figsize=(CONVERGENCE_FIG_WIDTH, CONVERGENCE_FIG_WIDTH / 4.0))
    x = dialogue.traces["P-understood"].keys()
    y = dialogue.traces["P-understood"].values()
    ax = gca()
    ax.plot(np.asarray(x) + 0.5, y, "o--", linewidth=2)
    ax.set_xticks(x[1:])
    xlim(x[0], x[-1] + 1)
    ylim(0, 1)
    ax.set_yticks([0, 0.5, 1.0])
    xlabel(TURN_XLABEL)
    ylabel("P(L understands S)")
    # These are in data coordinates
    IDEAL_BOX_WIDTH = 0.8
    BOX_BOTTOM_MARGIN = 0.05
    BOX_MIDDLE_MARGIN = 0.025
    IDEAL_BOX_HEIGHT = (0.5 - 2 * BOX_BOTTOM_MARGIN - BOX_MIDDLE_MARGIN) / 2
    # Boxes get centered between i and i+1 on the x axis
    # with VERT_MARGIN space below and BOX_MIDDLE_MARGIN between them
    # and their width/height is whichever of their IDEAL width/height is physically smaller
    ax2disp = ax.transData.transform
    ideal_box_size_display = ax2disp((IDEAL_BOX_WIDTH, IDEAL_BOX_HEIGHT)) - ax2disp((0, 0))
    box_size_display = min(ideal_box_size_display)
    disp2ax = ax.transData.inverted().transform
    box_width_data = (disp2ax((box_size_display, 0)) - disp2ax((0, 0)))[0]
    disp2fig = fig.transFigure.inverted().transform
    box_size_fig = disp2fig((box_size_display, box_size_display)) - disp2fig((0, 0))
    def ax2fig(xy):
        return disp2fig(ax2disp(xy))
    box_bottom_fig = ax2fig((0, BOX_BOTTOM_MARGIN))[1]
    box_middle_margin_fig = (ax2fig((0, BOX_MIDDLE_MARGIN)) - ax2fig((0, 0)))[1]
    box_vert_step_fig = box_middle_margin_fig + box_size_fig[1]
    def fig_left_of_box(box_i):
        return (ax2fig((box_i, 0))[0] + ax2fig((box_i + 1, 0))[0]) / 2 - box_size_fig[0] / 2
    for side_i, side in enumerate(["speaker", "listener"]):
        key = side + "-single-word-meanings"
        turns = dialogue.traces[key].keys()
        mats = dialogue.traces[key].values()
        for i, (turn, mat) in enumerate(zip(turns, mats)):
            sub_ax = fig.add_axes((fig_left_of_box(i),
                                   box_bottom_fig + (box_vert_step_fig) * (1 - side_i),
                                   box_size_fig[0],
                                   box_size_fig[1]))
            sub_ax.matshow(mat, cmap="binary", aspect="auto", vmin=0, vmax=1)
            if i == 0 and side_i == 1:
                sub_ax.set_ylabel("words")
                sub_ax.set_xlabel("objects")
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
    ax.set_zorder(10)
    ax.patch.set_alpha(0)
    fig.savefig(path, bbox_inches="tight")

def show_dialogues(base_path, dialogues, **kwargs):
    for i, dialogue in enumerate(dialogues):
        show_dialogue(base_path % (i,), dialogue, **kwargs)
