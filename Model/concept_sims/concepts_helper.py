## Define graphic functions and helper functions for concept simulations

# import stuff
import conpact2_C
import numpy as np
import matplotlib
import seaborn as sns
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import colors

# define constants
SOFTMAX_WEIGHT = 3
LISTENER_DEPTH = 2

# define helper functions

# Use a fixed max_utterance_length and softmax weight for everything
# With length-1 utterances we could get away with using actual sampling for posteriors...
def dom(**kwargs):
    return conpact2_C.MemoizedDomain(max_utterance_length=1, softmax_weight=SOFTMAX_WEIGHT, **kwargs)

# takes in a set of known maps between words and concepts (in terms of features), and returns biased lexicon prior in log space
def weight_lexicon_prior(words_to_concepts, lexicons, conceptsA, weight = 4):
    lexicon_prior = np.ones(len(lexicons))

    for km in range (0, len(words_to_concepts)):
        word = words_to_concepts[km][0]
        concept = words_to_concepts[km][1]
        concept_i = np.all(conceptsA == [concept], axis=1).nonzero()[0][0]
        for lex in range(0, len(lexicons)):
            if (lexicons[lex][word] == concept_i):
                lexicon_prior[lex] = weight

    lexicon_prior = np.log(lexicon_prior)           
    return(lexicon_prior)

# takes in word and objects (in terms of features), and returns lexicons with weight on lexicons consistent with both objects
# Used for antenna simulations.
def weight_lex_prior_given_obj(word, objects_data,objectsA, o_by_c, lexicons, weight = 4):
    object_is = [1] * len(objects_data)

    # get object indexes
    for o in range (0, len(objects_data)):
        object_is[o] = np.all(objectsA == [objects_data[o]], axis=1).nonzero()[0][0]

    # get concepts consistent with all objects
    concept_is = np.all(o_by_c[object_is,:] == 1, axis = 0).nonzero()[0]

    # weight those lexicons conistent with both
    lexicon_prior = np.ones(len(lexicons))
    for c in range (0, len(concept_is)):
        concept_i = concept_is[c]
        for lex in range(0, len(lexicons)):
            if (lexicons[lex][word] == concept_i):
                lexicon_prior[lex] = weight

    lexicon_prior = np.log(lexicon_prior)           
    return(lexicon_prior)


# takes in a set of objects, and returns biased object prior in log space
def weight_object_prior(weighted_objects, objectsA, weight = 4, zero_weight = 1e-10):
    object_prior = np.ones(len(objectsA))
    object_prior *= zero_weight

    for km in range (0, len(weighted_objects)):
        obj = weighted_objects[km]
        object_i = np.all(objectsA == [obj], axis=1).nonzero()[0][0]
        object_prior[object_i] = weight

    object_prior = np.log(object_prior)     
    return(object_prior)


## define graphic functions
# takes in lexicon_dist in log space and returns plot in normal space 
def show_lexicon_dist(lexicons, lexicon_weights, title = 'Distribution over lexicons', trim = 0):
    mask = (lexicon_weights > np.log(trim))
    lexicons = lexicons[mask, :]
    lexicon_weights = lexicon_weights[mask]
    num_lexs = np.shape(lexicons)[0]
    fig, ax = plt.subplots()
    
    # set up axes
    ax.set_xlim(0,num_lexs+1)
    #ax.set_ylim(0,1)
    ax.set_xlabel('lexicons')
    xticks(np.arange(num_lexs)+1)

    # draw plot
    pos = np.arange(num_lexs) + .6
    ax.bar(pos, np.exp(lexicon_weights), color=sns.husl_palette(6, s=.75), ecolor="#333333");
    ax.set_xticklabels(lexicons)
    fig.suptitle(title, fontsize=20)

# takes in object in log space and returns plot in normal space 
def show_object_dist(marginal_dist_n, utterance, objects, title = 'Distribution over objects in-the-moment, given utterance '):
    num_objects = np.shape(marginal_dist_n)[1]
    fig, ax = plt.subplots()

    # set up axes        
    ax.set_xlim(0, num_objects + 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('objects')
    xticks(np.arange(num_objects)+1)
    #ax.set_xticklabels(np.arange(num_objects))
    ax.set_xticklabels(objects)

    # draw plot
    pos = np.arange(num_objects) + .6
    ax.bar(pos, np.exp(marginal_dist_n[utterance[0], :]), color=sns.husl_palette(6, s=.75), ecolor="#333333");
    fig.suptitle('Distribution over objects in-the-moment, given utterance %d ' % utterance, fontsize=20)

# show utterances observed by learner
def show_utterances(utterances):
    num_utts = np.shape(utterances)[0]
    num_words = np.shape(utterances)[1] 
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5,.5,.5,1.0)
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    fig, ax = plt.subplots()
    fig.suptitle('Listener Data', fontsize=20)
    bounds = np.linspace(0,1,num_utts + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    pcolor(utterances, cmap=cmap)
    xticks(np.arange(num_words)+.5)
    yticks(np.arange(num_utts)+.5)
    ax.set_xticklabels(np.arange(num_words)+1)
    ax.set_yticklabels(np.arange(num_utts)+1)


    plt.xlabel('word slots')
    plt.ylabel('data points (utterances)')
    ax2 = plt.axes([0.95, 0.1, 0.03, 0.8])
    cb = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, spacing='proportional', norm = norm,
                                   ticks = np.arange(num_words+1)+.5, boundaries=bounds, format='%1i')
    ax2.set_ylabel('word values', size=12)
        

              

