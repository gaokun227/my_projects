# Solution for the Tech Screener for ETS AI Research Labs By Shuwen Zhang 

## Repository structure

* `README.md` - the description of the task.

* `sentences.csv` - the input from the programming task, which is a comma-separated file containing one sentence per line.

* `README_SOLUTION.md` - the description of the codes by Shuwen Zhang  (this file).

* `solution_nltk.py` - the python script that contains the necessary functions that extract the features from the given sets of sentences in `sentences.csv`.

* `test_solution_nltk.py` - the python script that test the main function in `solution_nltk.py` with various possible scenarios. 

## Overview of solution_nltk.py

This python script contains all the functions that are needed to fulfill the requirements of this programing task.

#### preprocess(sentence)

  * the function to correct typos in a given sentence

#### get_features(sentence_list)

  *  the function to extract features associated with prepositions from a given sentence

  *  input: a list of sentences;
            e.g., ['I am sentence1', 'I am sentence2', ...]

  *  return: a list of json objects with features extracted from each sentence

#### perform_required_task():

  * the function to read in the sentences.csv data and obtain the 12 required features

#### main()

   * the main function simply calls perform_required_task() and print out all the extracted features
    
## Sample Outputs from test_solution_nltk.py

This script test out the get_features() function in `solution_nltk.py` with some good or bad inputs. Here are some sample outputs returned from get_features().

#### A sentence with the word "to" followed by a verb
 * input: ["I am going to buy a car"]
 * the result is simply an empty list since there is no preposition in the sentence.

#### A very short sentence with only three words
 * input: ["live with love"]
 * the result is as follows:


```json
{"id":"1_1",
 "prep": "with",
 "features": ["live with", "with love", "live with love", "n/a live with", "with love n/a", "n/a live with love n/a", "JJ IN", "IN NN", "JJ IN NN", "N/A JJ IN", "IN NN N/A", "N/A JJ IN NN N/A"]]
}
```

* *Note* some features are padd if there are not enough elements: the w features are padded with 'n/a' and the t features are padded with 'N/A'.

#### The input is not a list variable

* input: i am not a list

* The function returns an empty list and print out the following error message

  "ERROR the input is not a list; check your input"

#### The input is a list but does not contain sentences

* input: [1, 2, 3]

* The function returns an empty list and print out the following error message

  "ERROR the input is not a sentence list; check your input"


## How to run the codes

* To run `solution_nltk.py`, do the following

  python `solution_nltk.py` 

* To run `test_solution_nltk.py`, do the following

  python `test_solution_nltk.py`

## Requirements
- [NLTK](https://www.nltk.org)
  * pip install nltk 
  * nltk.download('punkt') 
  * nltk.download('averaged_perceptron_tagger') 

- [PANDAS](https://pandas.pydata.org/)
  * pip install pandas 

## Potential areas for improvement

* Typos

  Not surprisingly there are typos in the sentences and sometimes the typo is the preposition itself (e.g., 'af' in sentence 11). It is desirable to correct these typos before the feature extraction step. At this point, I've only implemented a simple solution to correct the typos I've noticed due to the time limitation. For further work, it make sense to implement the spelling correction capacity in NLTK.

* Punctuation marks

  In my solution I did not remove the punctuation marks from the sentence before extracting the features as doing so can change the location of the preposition and therefore the 12 token and tag features. If the punctuation marks should not be included in the features (not clarified in the task description), I could add a preprocessing step to remove all of them in the sentences. 