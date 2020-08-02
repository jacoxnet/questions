import nltk
import sys
import os
import math
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # initialize dict
    files = {}
    # get file list
    file_list = os.listdir(directory)
    # iterate through file list reading in documents and
    # saving in dict with key=filename
    for file_name in file_list:
        with open(os.path.join(directory, file_name), encoding='utf8') as f:
            files[file_name] = f.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    word_list = []
    for word in nltk.word_tokenize(document):
        # don't include stop words at all
        new_word = word.lower()
        if new_word in nltk.corpus.stopwords.words("english"):
            continue
        new_word = "".join(c for c in new_word if c not in string.punctuation)
        if new_word == "":
            continue
        word_list.append(new_word)
    return word_list


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # initialize dict to hold counter
    count_words = {}
    # iterate through each document
    for doc in documents:
        # iterate through each unique word in doc
        for word in set(documents[doc]):
            # increment counter to show word found in doc
            if word in count_words:
                count_words[word] += 1
            else:
                count_words[word] = 1

    # define idf calcuation function for a word in count_words
    def idf_calc(x):
        return math.log(float(len(documents)) / count_words[x])
        
    idf = {word: idf_calc(word) for word in count_words}
    return idf


def calc_tfidf(query, wordlist, idfs):
    '''
    calculate and return the tf-idf for a query (a set of words), a wordlist
    (a list of words), and idfs as a dict wtih idfs vals by word
    '''
    found_qwords = [w for w in query if w in wordlist]
    tfidf = sum([wordlist.count(w) * idfs[w] for w in found_qwords if w in idfs])
    return tfidf


def calc_denidf(query, wordlist, idfs):
    '''
    Calculate and return a tuple with the density and the idf sum
    for a query (a set of words), a wordlist (a list of words),
    and idfs as a dict wtih idfs vals by word
    '''
    found_qwords = [w for w in query if w in wordlist]
    idf = sum([idfs[w] for w in found_qwords if w in idfs])
    density = len(found_qwords) / len(wordlist)
    return (idf, density)


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # create dict with filename key and tfidf value
    tfidf_vals = {f: calc_tfidf(query, files[f], idfs) for f in files}
    # sort by value
    sorted_tfidf_vals = sorted(tfidf_vals, key=lambda f: tfidf_vals[f], reverse=True)
    # return n top values
    return sorted_tfidf_vals[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # create dict with sentence key and (density, idf) as value
    senVals = {s: calc_denidf(query, sentences[s], idfs) for s in sentences}
    # sort by secondary key density
    secondVal = sorted(senVals, key=lambda s: senVals[s][1], reverse=True)
    # now sort by primary key idf
    primaryVal = sorted(secondVal, key=lambda s: senVals[s][0], reverse=True)
    # uncomment for debugging to add I and D values to sentence
    # for i in range(n):
    #    s = primaryVal[i]
    #    primaryVal[i] += f' (I {senVals[s][0]} D {senVals[s][1]})'
    return primaryVal[:n]


if __name__ == "__main__":
    main()
