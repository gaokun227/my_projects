*** PART 1: CODE DESCRIPTIONS

1. indicators.py: contains functions that calculate the technical indicators.

2. marketsimcode.py: a market simulator that accepts trading orders and keep track of a portfolio’s value over time.

3. RTLearner.py: a random tree learner object

4. BagLearner.py: a bag learner that ensembles the random tree learners  

5. ManualStrategy.py: implements the Manual Strategy object. It implements "testPolicy" which returns a trade data frame based on the selected stock symbol and date range.

6. StrategyLearner.py: implements the Strategy Learner object. It accepts training data through "add_evidence" to train the learner, and implements "testPolicy" to return a trade data frame.

7. experiment1.py: conducts experiment 1 as required by the project and generate the figures and statistics.

8. experiment2.py: conducts experiment 2 as required by the project and generate the figures and statistics.

9. testproject.py: runs all necessary codes that generate the figures and portfolio statistics for the report


*** PART2: HOW TO RUN THE CODES

1. To test all codes, run the following:
PYTHONPATH=../:. python testproject.py

2. To test ManualStrategy.py, run the following:
PYTHONPATH=../:. python ManualStrategy.py

3. To test StrategyLearner.py, run the following:
PYTHONPATH=../:. python StrategyLearner.py

4. To test experiment1.py, run the following:
PYTHONPATH=../:. python experiment1.py

5. To test experiment2.py, run the following:
PYTHONPATH=../:. python experiment2.py