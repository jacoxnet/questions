import nltk
import sys, os, math, string

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
    word_list = [
        word.lower() for word in nltk.word_tokenize(document) if (
            word not in string.punctuation and 
            word not in nltk.corpus.stopwords.words("english")
        )
    ]
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
                count_words[word] = 0
    # define idf calcuation function for a word in count_words
    idf_calc = lambda x: math.log(float(len(documents) / len(count_words[x])))
    idf = {word: idf_calc(word) for word in count_words
    return idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # initialize dict to collect sums
    tfidf_sums = {}
    # iterate over files
    for filename in files:
        # collect partial sums for each query word
        tfidf_sums[filename] = 0
        for qword in query:
            # add in count of qword in file times idfs
            # note that no guarantee there's an idfs entry for query word
            try:
                tfidf_sums[filename] += wordlist.count(qword) * idfs[qword]
            except KeyError:
                pass
    return sorted(tfidf_sums, key=lambda x: tfidf_sums[x], reverse=True)[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # initialize dict to collect sums
    idfs_sums = {}
    density_sums = {}
    # iterate over sentences
    for sentence, senwords in sentences.items():
        idfs_sums[sentence] = 0.0
        density_sums[sentence] = 0.0
        for qword in query:
            count = float(senwords.count(qword))
            density_sums[sentence] += count / len(senwords)
            try:
                idfs_sums[sentence] += idfs[qword]
            except KeyError:
                pass
    # return sorted primary by idfs secondary by density
    returnVal = sorted(idfs_sums, key=lambda x: density_sums[x], reverse=True)
    returnVal = sorted(returnVal, key=lambda x: idfs_sums[x], reverse=True)[:n]
    return returnVal
    

if __name__ == "__main__":
    main()
