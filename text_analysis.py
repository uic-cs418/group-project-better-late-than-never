import numpy as np
import cudf.pandas

cudf.pandas.install()
import pandas as pd
import nltk
from cuml.accel import install

install()
from cuml.svm import SVC
import sklearn
import string
import warnings
import re
from scipy import sparse
import pickle
import data_cleaning as dc
import review_score_analysis as rs


## Text Processing
def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    posMapping = {
        # "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
        "N": "n",
        "V": "v",
        "J": "a",
        "R": "r",
    }

    # Create regex to catch URLs
    url_regex = re.compile(
        r"""(
        (?:https?://)?        ## Optionally match http:// or https://
        (?:www\.)?            ## Optionally match www.
        [\w.-]+\.\w+          ## Match multiple domains (example.com or sub.domain.co.uk)
        (?:[/?#][^\s]*)?      ## Optionally match paths, queries, or fragments
    )""",
        re.VERBOSE,
    )

    ### Process string
    # Remove URLs
    text = url_regex.sub("", text).strip()
    # Remove all ('s) e.g. she's -> she
    text = re.sub("'s", "", text).strip()
    # Omit other apostrophes e.g. don't -> dont
    text = re.sub("'", "", text).strip()
    # swap all other punctuation with ' '
    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    # Set to lowercase
    text = str.lower(text)

    ### Process tokens
    # tokenize string
    tokenized_text = nltk.word_tokenize(text)
    # Tag tokens
    tokenized_text = nltk.pos_tag(tokenized_text)
    # lemmatize tokens, converting pos tags based on mappings above
    lemmatized_tokens = []
    for word, tag in tokenized_text:
        try:
            lemma = lemmatizer.lemmatize(word, pos=posMapping[tag[0]])
        except KeyError:
            # Anything not caught by posMapping dict has pos 'n'
            lemma = lemmatizer.lemmatize(word, pos="n")
        # except:
        #     # Ignore other exceptions
        #     continue
        lemmatized_tokens.append(lemma)

    return lemmatized_tokens


def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """process all text in the dataframe using process() function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process() function. Other columns are unaffected.
    """
    # Copy df to preserve original data
    processed_df = df.copy(deep=True)

    # Return df copy with processed text column
    processed_df["text"] = processed_df["text"].apply(process)
    return df


### Feature/Label Construction
def create_features(processed_reviews, stop_words):
    """creates the feature matrix using the processed review text
    Inputs:
        processed_reviews: pd.DataFrame: processed reviews read from train/test  file, containing the column 'text'
        stop_words: list(str): stop_words by nltk stopwords (after processing)
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
            we need this to tranform test reviews in the same way as train reviews
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    # Convert processed tweets text values to list of strings, with one tweet per string
    reviews_list = processed_reviews["text"].apply(lambda x: " ".join(x)).tolist()

    # Learn vocabulary and idf, return document-term matrix
    tfidf = sklearn.feature_extraction.text.TfidfVectorizer(
        min_df=2, lowercase=False, stop_words=stop_words
    )
    X = tfidf.fit_transform(reviews_list)

    return tfidf, X


def create_binary_labels(avg_scores_df):
    """
    creates two class labels from avg_review_score
    Inputs:
        avg_scores_df: pd.DataFrame: reviews read from training df, containing the column 'stars'
    Outputs:
        numpy.ndarray(int): series of class labels
        1 for restaurants with stars >= 4
        0 otherwise
    """
    # Apply vectorized  operation to score restaurants
    label_series = (avg_scores_df["stars"] >= 4).astype(int)

    return label_series


def create_3_labels(avg_scores_df):
    """
    creates three class labels from avg_review_score
    Inputs:
        avg_scores_df: pd.DataFrame: reviews read from training df, containing the column 'stars'
    Outputs:
        numpy.ndarray(int): series of class labels
        2 for restaurants with avg_review_score == 5
        1 for restaurants with stars == 4
        0 otherwise
    """

    def classify(score):
        if score < 4:
            return 0
        elif score == 4:
            return 1
        else:
            return 2

    label_series = avg_scores_df["stars"].apply(classify)

    return label_series


### Classification
def learn_classifier(X_train, y_train, kernel):
    """learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
        kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
    Outputs:
        sklearn.svm.SVC: classifier learnt from data
    """

    classifier = SVC(kernel=kernel)
    classifier.fit(X_train, y_train)

    return classifier


def train_binary_model(avg_scores_df, size):
    """
    Brings it all together
    1. Creates a TFIDF feature matrix from the text of yelp reviews
    2. Creates labels for those features
    4. Trains a Support Vector Classifier (SVC) to classify if a review came from
       a restaurant with stars >= 4, or not. Uses {size} datapoints
    5. Returns X, y, tfidf, SVC
    """
    # Create features
    processed_reviews = process_all(avg_scores_df.loc[0:size])
    stopwords = nltk.corpus.stopwords.words("english")
    processed_stopwords = list(np.concatenate([process(word) for word in stopwords]))
    (tfidf, X) = create_features(processed_reviews, processed_stopwords)

    # Create labels
    y = create_binary_labels(avg_scores_df.loc[0:size])

    # Train model
    review_classifier = learn_classifier(X, y, "linear")

    return X, y, tfidf, review_classifier


def train_3_class_model(avg_scores_df, size):
    """
    Trains a model to identify a review as bad, good, or great
    1. Creates a TFIDF feature matrix from the text of yelp reviews
    2. Creates labels for those features
    4. Trains a Support Vector Classifier (SVC) to classify if a review came from
       a restaurant with score < 4, score == 4, or score == 5
    5. Returns X, y, tfidf, SVC
    """
    # Create features
    processed_reviews = process_all(avg_scores_df.loc[0:size])
    stopwords = nltk.corpus.stopwords.words("english")
    processed_stopwords = list(np.concatenate([process(word) for word in stopwords]))
    (tfidf, X) = create_features(processed_reviews, processed_stopwords)

    # Create labels
    y = create_3_labels(avg_scores_df.loc[0:size])

    # Train model (cuml can only do binary SVC)
    review_classifier = sklearn.svm.SVC(kernel="linear", decision_function_shape="ovr")
    review_classifier.fit(X, y)

    return X, y, tfidf, review_classifier


## Evaluation
def create_binary_test_data(reviews_df, size, tfidf):
    """
    Creates test data with 'size' datapoints, to be evaluated by trained model
    1. Creates a TFIDF feature matrix from the text of yelp reviews (needs training tfidf)
    2. Creates labels for those features
    5. Returns X (test_features), y (test_labels)
    """
    # Create features with data points not used in training
    processed_reviews = process_all(reviews_df.loc[1_000_000 : (size + 1_000_000)])
    tfidf_input = processed_reviews["text"].apply(lambda x: " ".join(x)).tolist()
    X = tfidf.transform(tfidf_input)

    # Create labels
    y = create_binary_labels(reviews_df.loc[1_000_000 : (size + 1_000_000)])

    return X, y


def create_multiclass_test_data(avg_scores_df, size, tfidf):
    """
    Creates test data with 'size' datapoints, to be evaluated by trained model
    1. Creates a TFIDF feature matrix from the text of yelp reviews (needs training tfidf)
    2. Creates labels for those features
    5. Returns X (test_features), y (test_labels)
    """
    # Create features with data points not used in training
    processed_reviews = process_all(avg_scores_df.loc[1_000_000 : (size + 1_000_000)])
    tfidf_input = processed_reviews["text"].apply(lambda x: " ".join(x)).tolist()
    X = tfidf.transform(tfidf_input)

    # Create labels
    y = create_3_labels(avg_scores_df.loc[1_000_000 : (size + 1_000_000)])

    return X, y


def evaluate_classifier(classifier, X_validation, y_validation):
    """evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.classes.SVC: classifer to evaluate
        X_validation: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_validation: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
    # Run classification
    predicted_labels = classifier.predict(X_validation)
    # Calculate accuracy of predictions
    accuracy = sklearn.metrics.accuracy_score(y_validation, predicted_labels)

    return accuracy


class MajorityLabelClassifier:
    """
    A classifier that predicts the mode of training labels
    """

    def __init__(self):
        """
        Initialize your parameter here
        """
        # Declare uninitialized mode
        self.mode = np.nan

    def fit(self, X, y):
        """
        Implement fit by taking training data X and their labels y and finding the mode of y
        i.e. store your learned parameter
        """
        # Convert y to a series, if it is not already
        y = pd.Series(y)

        # Count number of values in each label
        counts = y.value_counts()
        # Set mode to index (i.e. label) of most frequently occuring value
        self.mode = counts.idxmax()

    def predict(self, X):
        """
        Implement to give the mode of training labels as a prediction for each data instance in X
        return labels
        """
        predicted_labels = []
        for value in X:
            predicted_labels.append(self.mode)

        return predicted_labels


def benchmark(X, y):
    """
    Creates a MajorityLabelClassifer to output the benchmark for our model to 'beat'
    """
    baselineClf = MajorityLabelClassifier()
    # Use fit and predict methods to get predictions and compare it with the true labels y
    baselineClf.fit(X, y)
    predicted_labels = baselineClf.predict(X)

    baseline = sklearn.metrics.accuracy_score(y, predicted_labels)
    print(f"Benchmark accuracy for our model to beat: {baseline}")


## Cross-validation
def binary_kernel_cross_validation(X, y):
    """
    Select the kernel giving best results using k-fold cross-validation.
    Other parameters should be left default.
    Input:
    kf (sklearn.model_selection.KFold): kf object defined above
    X (scipy.sparse.csr.csr_matrix): training data
    y (array(int)): training labels
    Return:
    best_kernel (string)
    """
    # Use dict to store results of each evaluation, initialize to NaN
    avg_kernel_accuracies = {
        "linear": np.nan,
        "rbf": np.nan,
        "poly": np.nan,
        "sigmoid": np.nan,
    }

    # Use 4-fold split
    kf = sklearn.model_selection.KFold(n_splits=4, random_state=1, shuffle=True)

    for kernel in ["linear", "rbf", "poly", "sigmoid"]:
        scores = []
        # Use the documentation of KFold cross-validation to split ..
        # training data and test data from create_features() and create_labels()
        for i, (training_split, test_split) in enumerate(kf.split(X, y)):
            # call learn_classifer() using training split of kth fold
            classifier = learn_classifier(X[training_split], y[training_split], kernel)
            # evaluate on the test split of kth fold
            accuracy = evaluate_classifier(classifier, X[test_split], y[test_split])
            print(f"Accuracy of {kernel} kernel on split {i}: {accuracy}")
            scores.append(accuracy)

        # record avg accuracies and determine best model (kernel)
        avg_kernel_accuracies[kernel] = np.average(scores)

    # return best kernel as string
    best_kernel = max(avg_kernel_accuracies, key=avg_kernel_accuracies.get)
    return best_kernel


def three_way_cross_validation(X, y):
    """
    Select the kernel giving best results using k-fold cross-validation.
    Other parameters should be left default.
    Input:
    kf (sklearn.model_selection.KFold): kf object defined above
    X (scipy.sparse.csr.csr_matrix): training data
    y (array(int)): training labels
    Return:
    best_kernel (string)
    """
    # Use dict to store results of each evaluation, initialize to NaN
    avg_kernel_accuracies = {
        "linear": np.nan,
        "rbf": np.nan,
        "poly": np.nan,
        "sigmoid": np.nan,
    }

    # Use 4-fold split
    kf = sklearn.model_selection.KFold(n_splits=4, random_state=1, shuffle=True)

    for kernel in ["linear", "rbf", "poly", "sigmoid"]:
        scores = []
        # Use the documentation of KFold cross-validation to split ..
        # training data and test data from create_features() and create_labels()
        for i, (training_split, test_split) in enumerate(kf.split(X, y)):
            # train classifier using training split of kth fold
            classifier = sklearn.svm.SVC(kernel=kernel, decision_function_shape="ovr")
            classifier.fit(X[training_split], y[training_split])
            # evaluate on the test split of kth fold
            accuracy = evaluate_classifier(classifier, X[test_split], y[test_split])
            print(f"Accuracy of {kernel} kernel on split {i}: {accuracy}")
            scores.append(accuracy)

        # record avg accuracies and determine best model (kernel)
        avg_kernel_accuracies[kernel] = np.average(scores)

    # return best kernel as string
    best_kernel = max(avg_kernel_accuracies, key=avg_kernel_accuracies.get)
    return best_kernel


## Save/Load
def save_model(features, labels, tfidf, classifier, model_name):
    """
    saves the features, labels, and classifier to individual files
    """
    with open(f"data/{model_name}_features.pkl", "wb") as file:
        pickle.dump(features, file)

    with open(f"data/{model_name}_labels.pkl", "wb") as file:
        pickle.dump(labels, file)

    with open(f"data/{model_name}_tfidf.pkl", "wb") as file:
        pickle.dump(tfidf, file)

    with open(f"data/{model_name}_classifier.pkl", "wb") as file:
        pickle.dump(classifier, file)

    return


def load_model(model_name):
    """
    loads the features, labels, and classifier into individual variables
    """
    with open(f"data/{model_name}_features.pkl", "rb") as features:
        X = pickle.load(features)

    with open(f"data/{model_name}_labels.pkl", "rb") as labels:
        y = pickle.load(labels)

    with open(f"data/{model_name}_tfidf.pkl", "rb") as tfidf:
        tfidf = pickle.load(tfidf)

    with open(f"data/{model_name}_classifier.pkl", "rb") as classifier:
        review_classifier = pickle.load(classifier)

    return X, y, tfidf, review_classifier
