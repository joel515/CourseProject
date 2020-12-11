import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import os
import glob
import re
import argparse
import datetime

# Substitution for infinitesimal numbers to avoid log(0) issues
ZERO = 1.7976931348623157e-323


def check_file_validity(parser: argparse.ArgumentParser, file: str) -> str:
    """
    Utility to check the existance of a file
    """
    if not os.path.isfile(file):
        parser.error(f"specified file does not exist ({file})")
    else:
        return file


def normalize(input_matrix: np.array) -> np.array:
    """
    Utility to normalize 2D or 3D numpy matrices
    """
    ndim = len(np.shape(input_matrix))
    if ndim == 2:
        return normalize_2d(input_matrix)
    elif ndim == 3:
        return normalize_3d(input_matrix)

    raise RuntimeError(f"{ndim} dimensional matrix normalization not supported")


def normalize_2d(input_matrix: np.array) -> np.array:
    """
    Utility to normalize 2D numpy matrices
    """
    inv_sum = np.nan_to_num(1/np.sum(input_matrix, axis=1))
    return np.einsum('ij,i->ij', input_matrix, inv_sum)


def normalize_3d(input_matrix: np.array) -> np.array:
    """
    Utility to normalize 3D numpy matrices
    """
    inv_sum = np.nan_to_num(1/np.sum(input_matrix, axis=1))
    return np.einsum('ijk,ik->ijk', input_matrix, inv_sum)

       
class Corpus(object):
    """
    A collection of documents
    """

    def __init__(self, documents_path):
        """
        Initialize variables
        """
        self.documents = []
        self.views = []
        self.view_text = []
        self.vocabulary = []
        self.likelihoods = []

        self.documents_path = documents_path
        self.data = None

        self.term_doc_matrix = None # c(w,D)
        self.coverage_dist = None   # P(l|k)
        self.topic_dist = None      # P(w|theta)
        self.view_prob = None       # P(v|D,C)
        self.latent_dist = None     # P(z|v,l,w)

        self.n_docs = 0
        self.n_views = 0
        self.n_words = 0


    def build_corpus(self, preprocessing):
        """
        Tokenize the documents into a bag of words and preprocess as requested
        Ideas for preprocessing techniques taken from https://github.com/yedivanseven/PLSA
        """
        punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
        ps = PorterStemmer()

        # Read in the text and metadata and pull out the text for preprocessing and tokenization
        self.data = pd.read_csv(self.documents_path, encoding="ISO-8859-1")
        documents = self.data["text"].values

        for doc in documents:
            # Remove non-ASCII characters
            if preprocessing["ascii"]:
                doc = "".join(char if ord(char) < 128 else " " for char in str(doc))

            # Convert all to lower case
            if preprocessing["lc"]:
                doc = str(doc).lower()

            # Remove all punctuation (except '-')
            if preprocessing["punc"]:
                doc = doc.translate(str.maketrans({str(char): " " for char in punctuation}))

            # Remove all numbers
            if preprocessing["num"]:
                doc = "".join(filter(lambda char: not char.isdigit(), str(doc)))

            # Tokenize
            words_in_doc = doc.split()

            # Stemming
            if preprocessing["stem"]:
                words_in_doc = [ps.stem(word) for word in words_in_doc]

            # Remove stop words
            if preprocessing["stop"]:
                words_in_doc = [word for word in words_in_doc if not word in stop_words]

            self.documents.append(words_in_doc)

        self.n_docs = len(self.documents)


    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus
        """
        vocabulary = []
        for doc in self.documents:
            vocabulary.extend(list(set(doc)))

        self.vocabulary = list(set(vocabulary))
        self.vocabulary.sort()
        self.n_words = len(self.vocabulary)


    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term
        """
        def fn(x, y):
            return self.documents[x].count(self.vocabulary[y])
        self.term_doc_matrix = np.fromfunction(np.vectorize(fn), (self.n_docs, self.n_words), dtype=int)


    def build_views(self, views):
        """
        Construct the context views as a bit array
        """
        print("Generating views from metadata...")

        # View text is just the Boolean operations strings set as arguments
        self.view_text = views
        self.n_views = len(views)

        # Variables will be the first word in each string (corresponding to columns in metadata)
        variables = list(set([re.search(r"^(\w+)", x).group(1) for x in views[1:]]))

        # Initialize to all zeros, then set global view flags to 1
        self.views = np.zeros([self.n_docs, self.n_views], dtype=int)
        self.views[:,0] = np.repeat([1], self.n_docs)

        # For each document, set the view flags accordingly using the metadata and Boolean operations
        # specified in the arguments
        for idx1, row in self.data.iterrows():
            for idx2, v in enumerate(views[1:]):
                for var in variables:
                    exec(f"{var} = {row[var]}")
                if eval(v):
                    self.views[idx1,idx2+1] = 1


    def initialize(self, n_topics, prior):
        """
        Randomly initialize the coverage distributions, topic distributions, and view probabilities (with normalization)
        """
        print("Initializing...")

        # Coverage distribution P(l|k) initialization
        self.coverage_dist = np.random.random_sample((self.n_docs, n_topics))
        self.coverage_dist = normalize(self.coverage_dist)

        # Topic distribution P(w|theta) initialization
        self.topic_dist = np.random.random_sample((n_topics, self.n_words, self.n_views))
        self.topic_dist = normalize(self.topic_dist)

        # View probability P(v|D,C) initialization
        self.view_prob = np.random.random_sample((self.n_docs, self.n_views))

        # Account for view features by setting probabilities not related to document to zero (using bit array)
        self.view_prob = np.einsum('ij,ij->ij', self.view_prob, self.views)

        # Set the global prior artificially high (based on prior arg) then normalize
        self.view_prob[:,0] = np.repeat([prior], self.n_docs)
        self.view_prob = normalize(self.view_prob)


    def expectation_step(self):
        """
        E-step to update P(z|v,l,w)
        """
        print("E step...")
        
        # Perform outer dot of view probabilities, coverage distributions, and topic distributions
        self.latent_dist = np.einsum('il,ij,jkl->ijkl', self.view_prob, self.coverage_dist, self.topic_dist)

        # Bayesian normalization - sum over topics then views
        latent_dist_sum = np.sum(np.einsum('ij,jkl->ijkl', self.coverage_dist, self.topic_dist), axis=1)
        latent_dist_sum = np.sum(np.einsum('ik,ijk->ijk', self.view_prob, latent_dist_sum), axis=2)
        inv_latent_dist_sum = np.nan_to_num(1/latent_dist_sum)
        self.latent_dist = np.einsum('ijkl,ik->ijkl', self.latent_dist, inv_latent_dist_sum)


    def maximization_step(self):
        """ 
        M-step to update P(v|D,C), P(l|k), and P(w|theta)
        """
        print("M step...")

        # Update topic distribution P(w|theta) and sum across the documents in the collection
        self.topic_dist = np.sum(np.einsum('ik,ijkl->ijkl', self.term_doc_matrix, self.latent_dist), axis=0)

        # Normalize across vocabulary
        self.topic_dist = normalize(self.topic_dist)

        # Update coverage distribution P(l|k) and sum across views then vocabulary
        self.coverage_dist =np.sum(np.einsum('ik,ijk->ijk', self.term_doc_matrix, np.sum(self.latent_dist, axis=3)), axis=2)

        # Normalize across topics
        self.coverage_dist = normalize(self.coverage_dist)

        # Update view probabilies P(v|D,C) and sum across topics then vocabulary
        self.view_prob = np.sum(np.einsum('ij,ijk->ijk', self.term_doc_matrix, np.sum(self.latent_dist, axis=1)), axis=1)

        # Normalize across views
        self.view_prob = normalize(self.view_prob)


    def calculate_likelihood(self) -> np.float64:
        """
        Calculate the current log-likelihood of the model using the model's updated probability matrices
        """
        # Product of coverage distribution (P(l|k)) and topic distribution (P(w|theta)), summed over topics
        cov_topic_sum = np.sum(np.einsum('ij,jkl->ijkl', self.coverage_dist, self.topic_dist), axis=1)

        # Product of previous sum and view distribution (P(v|D,C)), summed over views
        view_cov_topic_sum = np.sum(np.einsum('ik,ijk->ijk', self.view_prob, cov_topic_sum), axis=2)

        # Assign infinitiesimal number for 0 for log calculations
        view_cov_topic_sum = np.where(view_cov_topic_sum==0, ZERO, view_cov_topic_sum)
        log_view_cov_topic_sum = np.log(view_cov_topic_sum)

        # Multiply log value times word count and sum over vocabulary and documents/contexts
        current_likelihood = np.sum(np.einsum('ij,ij->ij', self.term_doc_matrix, log_view_cov_topic_sum))
        self.likelihoods.append(current_likelihood)
        
        # Return the current log-likelihood value
        return current_likelihood


    def cplsa(self, n_topics: int, max_iter: int, epsilon: float, thresh: float, prior: float, warmup: bool):
        """
        Expectation-Maximization iterations for CPLSA mixture model

        :n_topics: Number of topics to model
        :max_iter: Maximum number of iterations if convergence is not met
        :epsilon: Convergence criterion for MLE
        :thresh: View probability threshold to stop warm up iteration 
        :prior: Artificial prior to assign to global views to increase importance
        :warmup: Boolean signifying if these will be warmup iterations
        """
        if warmup:
            print ("EM warmup iterations begin...")
            
            # Build the term-document matrix
            self.build_term_doc_matrix()
            
            # Initialize the latent probability as zeros (P(z|v,l,w))
            self.latent_dist = np.zeros([self.n_docs, n_topics, self.n_words, self.n_views], dtype=np.double)

            # Initialize the components of the latent probability (P(v|D,C), P(l|k), P(w|theta))
            self.initialize(n_topics, prior)

            current_likelihood = 0.0
            current_view_mean = 1.0
        else:
            print ("EM iterations restart...")

            # If restarting, get the last likelihood and calculate the current view probability mean for the global view
            current_likelihood = self.likelihoods[-1]
            current_view_mean = np.sum(self.view_prob, axis=0)[0]/self.n_docs

        for iteration in range(max_iter):
            print(f"Iteration ({str(iteration + 1)}/{str(max_iter)})...")

            # Run the expectation step
            self.expectation_step()

            # Run the maximization step
            self.maximization_step()

            # Calculate the new likelihood and compare it to the old value to determine convergence
            likelihood = self.calculate_likelihood()
            likelihood_delta = abs(likelihood - current_likelihood)

            # Calculate the new view probability mean for the global view to see if that's converges (for warmup)
            view_mean = np.sum(self.view_prob, axis=0)[0]/self.n_docs
            view_mean_delta = abs(current_view_mean - view_mean)

            current_view_mean = view_mean
            print("Mean prob. of global view (delta): %s (%s)" % (str(current_view_mean), view_mean_delta))
            
            current_likelihood = likelihood
            print("Likelihood (delta): %s (%s)" % (current_likelihood, likelihood_delta))

            if warmup:
                # If we are in the warmup phase and the global view mean probability is below the
                # threshold, stop iterating
                if view_mean < thresh:
                    break
            else:
                # If the MLE has converged, stop iterating
                if likelihood_delta < epsilon:
                    break

        # Write out the coverages
        if not warmup:
            outfile = f"CPLSA-{datetime.datetime.now().isoformat()}.out"
            outfile = outfile.replace(":", "-")
            print(f"Writing {outfile}")
            with open(outfile, "w") as f:
                f.write(f"***** Results of CPLSA analysis for {self.documents_path} *****\n")
                f.write(f"Maximum Likelihood Estimate: {current_likelihood}\n")
                f.write(f"Iterations: {str(iteration+1)} out of {max_iter}\n")
                f.write(f"Topic count: {n_topics}\n")
                f.write(f"View count: {self.n_views}\n")
                f.write(f"Document count: {self.n_docs}\n")
                f.write(f"Vocabulary size: {self.n_words}\n")
                f.write(f"Initial global view prior: {prior}\n\n")

                f.write(f"***** Coverages *****\n\n")

                for topic in range(0, n_topics):
                    for view in range(0, self.n_views):
                        f.write("\n")
                        f.write("Topic: " + str(topic+1) + ", View: " + self.view_text[view] + "\n")
                        idx = np.argsort(self.topic_dist[topic,:,view])[::-1]
                        for i in idx[:10]:
                            f.write(self.vocabulary[i] + "\t" + str(self.topic_dist[topic][i][view]) + "\n")


def main(args: argparse.ArgumentParser, views: list, pkl=None):
    """
    Main implementation of CPLSA utility

    :args: Argument namespace from argparse
    :views: List of boolean operations representing the contextual views
    :pkl: Pickle file for restart
    """
    # If a pickle file is supplied, then we are restarting - no need to initialize
    if pkl:
        with open(pkl, "rb") as f:
            corpus = pickle.load(f)
        os.remove(pkl)
        warmup = False
        max_iter = args.iterations
    else:
        warmup = True
        max_iter = args.warmup_iter
        preprocessing = {
            "ascii": not args.noASCII,
            "lc": not args.noLC,
            "punc": not args.noPUNC,
            "num": not args.noNUM,
            "stem": not args.noSTEM,
            "stop": not args.noSTOP,
        }

        # Initialize our corpus, preprocess, and build the vocabulary and views
        corpus = Corpus(args.file)
        corpus.build_corpus(preprocessing)
        corpus.build_vocabulary()
        corpus.build_views(views)

    print("Vocabulary size:" + str(corpus.n_words))
    print("Number of documents:" + str(corpus.n_docs))
    print("Number of views:" + str(corpus.n_views))

    # Process the corpus through CPLSA
    corpus.cplsa(args.topics, max_iter, args.epsilon, args.threshold, args.prior, warmup)

    write = args.save
    if warmup:
        filepfx = "corpus"

        # If we are in the initialization phase, calculate the mean view probability - if it's below our
        # threshold, pickle the Corpus instantiation to restart later (if it has the lowest MLE)
        view_mean = np.sum(corpus.view_prob, axis=0)[0]/corpus.n_docs
        if view_mean < args.threshold:
            print("View probability mean below threshold...")
            current_likelihood = int(corpus.likelihoods[-1])
            for f in glob.glob("corpus-*"):
                m = re.search(r"corpus-(\d+)", f)
                if m:
                    ml = -int(m.group(1))
                    if current_likelihood < ml:
                        write = False
                    else:
                        os.remove(f)
                        write = True
                else:
                    write = False
        else:
            # If the view probability is not below the threshold, don't save it
            print("View probability mean converged prior to threshold")
            write = False
    else:
        filepfx = "final-corpus"

    # Pickle the Corpus, if appropriate/requested
    if write:
        pkl = f"{filepfx}{str(int(corpus.likelihoods[-1]))}.pkl"
        print(f"Writing Corpus to pickle: {pkl}")
        with open(pkl, "wb") as f:
            pickle.dump(corpus, f)


def parse_views(meta1: str, meta2: str) -> list:
    """
    Utility to parse the metadata in the argument strings to formulate views
    """
    # Each Boolean operation should be separated by ':' in the 2 strings
    meta1 = meta1.split(":")
    meta2 = meta2.split(":")

    # First view will be the global view
    views = ["global"]

    # Add the individual views as dictated by the input strings
    views.extend(meta1)
    views.extend(meta2)

    # Add the different combinations of individual views to form a new Boolean operation
    for m1 in meta1:
        for m2 in meta2:
            views.append(f"{m1} and {m2}")

    # Print out the discovered views
    n_views = len(views)
    print(f"There were {n_views} views discovered:")
    for f in views:
        print(f" -{f}")
    print()

    return views


def _parse_args() -> argparse.ArgumentParser:
    """
    Parse arguments using argparse
    """
    parser = argparse.ArgumentParser(
        description="Perform Contextual Probabilistic Semantic Analysis on text and metadata",
        prog="cplsa.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="All preprocessing is on by default: stopword, non-ASCII, number, and punctuation (except '-') removal, lower case transformation, and Porter stemming")

    parser.add_argument("file", type=lambda x: check_file_validity(parser, x), help="labeled input text and metadata CSV file")
    parser.add_argument("meta1", type=str, help="string of evaluatable boolean operators for metadata set 1 (example \"author==1:author==2\")")
    parser.add_argument("meta2", type=str, help="string of evaluatable boolean operators for metadata set 2 (example \"year<=1992:year>=1993 and year<=1999:year>=2000\")")
    parser.add_argument("-w", "--warmup", type=int, default=20, help="number of initial warmup runs for initialization")
    parser.add_argument("-p", "--prior", type=float, default=1.0, help="prior to assign global theme during warmup")
    parser.add_argument("-th", "--threshold", type=float, default=0.1, help="threshold on global view to cease warmup")
    parser.add_argument("-wi", "--warmup_iter", type=int, default=25, help="maximum number of warmup E-M iterations")
    parser.add_argument("-t", "--topics", type=int, default=10, help="number of topics/themes")
    parser.add_argument("-i", "--iterations", type=int, default=1000, help="maximum number of E-M iterations")
    parser.add_argument("-e", "--epsilon", type=float, default=0.001, help="maximum likelihood estimate error for convergence")
    parser.add_argument("-s", "--save", help="pickle the final results", action="store_true")

    parser.add_argument("-noASCII", "--noASCII", help="switch off non-ASCII character removal", action="store_true")
    parser.add_argument("-noLC", "--noLC", help="switch off lower case transformation", action="store_true")
    parser.add_argument("-noPUNC", "--noPUNC", help="switch off punctuation removal", action="store_true")
    parser.add_argument("-noNUM", "--noNUM", help="switch off number removal", action="store_true")
    parser.add_argument("-noSTEM", "--noSTEM", help="switch off Porter stemming", action="store_true")
    parser.add_argument("-noSTOP", "--noSTOP", help="switch off stopword removal", action="store_true")

    args = parser.parse_args()

    return args


def _report_args(args: argparse.ArgumentParser):
    """
    Utility to print out options
    """
    print(f"Analyzing contextual PLSA for {args.file}...")
    print(f"  --warmup      {args.warmup}\tnumber of initial warmup runs for initialization")
    print(f"  --prior       {args.prior}\tprior to assign global theme during warmup")
    print(f"  --threshold   {args.threshold}\tthreshold on global view to cease warmup")
    print(f"  --warmup_iter {args.warmup_iter}\tmaximum number of warmup E-M iterations")
    print(f"  --topics      {args.topics}\tnumber of topics/themes")
    print(f"  --iterations  {args.iterations}\tmaximum number of E-M iterations")
    print(f"  --epsilon     {args.epsilon}\tmaximum likelihood estimate error for convergence")
    print(f"  Preprocessing directives:")
    if not args.noSTOP:
        print("    Stopword removal")
    if not args.noSTEM:
        print("    Porter stemming")
    if not args.noNUM:
        print("    Number removal")
    if not args.noPUNC:
        print("    Punctuation removal (except '-')")
    if not args.noLC:
        print("    Lower-case transformation")
    if not args.noASCII:
        print("    Non-ASCII character removal")
    print()


if __name__ == '__main__':
    args = _parse_args()

    # Get the absolute path for the input CSV file
    args.file = os.path.abspath(args.file)

    # Report out all options
    _report_args(args)

    # Formulate the views using the Boolean specification arguments
    views = parse_views(args.meta1, args.meta2)

    # First process the warmup loops to find the best starting point
    print(f"Running {args.warmup} initialization loop(s)...")
    for i in range(0, args.warmup):
        msg = f"Warmup loop {str(i+1)} of {str(args.warmup)}"
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))
        main(args, views)

    # Get the pickled Corpus with the lowest MLE and restart from there
    # until convergence (should only be 1 file)
    max_ml = 1000000
    for f in glob.glob("corpus-*"):
        m = re.search(r"corpus-(\d+)", f)
        if m:
            ml = int(m.group(1))
            if ml < max_ml:
                max_ml = ml

    pkl = f"corpus-{str(max_ml)}.pkl"
    msg = f"Restarting from {pkl}"
    print()
    print("*"*len(msg))
    print(msg)
    print("*"*len(msg))
    print()
    main(args, views, pkl)
