from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

def main():

    '''
        dict: [{'orig': 'sentence', 'synsets': 'synsets'}]
    '''

    # Filter for Nouns, Verbs, Adjectives and Adverbs
    POS_WHITELIST = [ 'NOUN', 'VERB', 'ADJ', 'ADV' ]
    # Dictionary to convert to wordnet POS constants
    WORDNET_POS = { 'NOUN': wordnet.NOUN,
                    'VERB': wordnet.VERB,
                    'ADJ': wordnet.ADJ,
                    'ADV': wordnet.ADV }

    with open(filename, 'r') as infile:
        lines = [ line for line in infile.readlines() ]

    token_lines = [ word_tokenize(i) for i in lines ]

    for token_sentence in token_lines:
        sentence_tags = pos_tag(token_sentence, tagset='universal')

        syn = []
        for token, tag in sentence_tags:
            if tag in POS_WHITELIST:
                syns = wordnet.synsets(token ,pos=WORDNET_POS[tag])
                if syns:
                    syn.append(syns[0])
            else:
                syn.append(tag)
        print(syn)

if __name__ == "__main__":
    main()
