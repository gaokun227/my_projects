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

This script test out the get_features() function in `solution_nltk.py` with some good or bad inputs. Here are some sample output returned from get_features().

#### A sentence with the word "to" followed by a verb.
 * input: ["I am going to buy a car"]
 * the result is simply an empty list since there is no preposition in the sentence.

#### A very short sentence with only three words
 * input: ["live with love"]
 * the result is as follows:


```json
{"id":"1_1",
 "prep": "with",
 "features": ['live with', 'with love', 'live with love', 'n/a live with', 'with love n/a', 'n/a live with love n/a', 'JJ IN', 'IN NN', 'JJ IN NN', 'N/A JJ IN', 'IN NN N/A', 'N/A JJ IN NN N/A']]
}
```

* Note some features are padd. The w features with 'n/a' and the t features with 'N/A'.

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




== do delete
Your task is to write Python code that extracts features that can then be used to train a hypothetical grammatical error correction system.

The code should meet the following requirements:

* It should tokenize sentences, tag them, and extract features for prepositions (see below for feature definitions). The input data consists of one sentence per line and is contained in the file `sentences.csv` in this repository.

* It should be written in Python 3 following standard coding conventions;

* It should have sufficient documentation for users and for other developers (both inline documentation as well as a README describing how to run the code);

Extra credit will be awarded to the submission that has appropriate unit and functional tests, wherever possible. You can choose your own preferred framework for unit testing and for documentation. If you do not have time to implement the tests, feel free to provide a short description of the kinds of test you might have implemented if you had more time. 

This screening task is designed to get a sense of the technical skills of candidates applying for engineering positions in the ETS AI Research Labs.

### Definition of prepositions for this task
The following set of tokens are considered to be prepositions:

- *on*
- *for*
- *of*
- *to*
- *at*
- *in*
- *with*
- *by*

[*Note*: "to" is not considered to be a preposition when followed by a verb.]

### Feature extraction

![feature extraction description latex image](feature_extraction_description.svg)

Each set of features should have an id that consists of the sentence number and the token number. So for example, if the second token in the first sentence is a preposition, it would have id "1_1".

### Output format
A file containing one output line per preposition consisting of a JSON object with the following structure:



### Example

For example, the output for the first preposition `in` which occurs at token position 6 in the the first sentence with id `1` should be the following JSON:



### Suggested Python libraries
Feel free to use any publicly available Python libraries for tokenizing and tagging the input sentences. Here are a couple of suggestions: 

- [NLTK](https://www.nltk.org)
- [Spacy](https://spacy.io)

## Submission instructions

1. Clone this repository. 

2. Create a new branch called "submission". Add your code, tests, and documentation to this branch. All files should be text-only. Do not use any binary formats. 

3. Once your code is complete, push your branch, and submit a pull request to have your branch merged into the `main` branch. Submit all your work to this repository. Do *not* create a new repository or any other external online resources. 

4. **Do not merge this pull request. Leave it open.**

## Important Notes

- Approach the task as if your were asked to write code with this functionality to be used by other developers working on the same grammar error correction system. In addition to documenting your code, you should provide a README documenting your solution and implementation.

- We expect that 24 hours are sufficient to write documented code, some basic documentation as well as some simple tests. 

- If you feel like you are running out of time, you can skip the implementation of the unit tests and just provide a short description of some sample tests you would have implemented if you had more time. If you have any comments about the task, you can include them with the documentation.
