# Project 1: Applied Statistical Concepts in Data Science

## Project Overview

This project explores fundamental statistical concepts through a series of three practical case studies. The goal is to build a strong foundation in data analysis, simulation, and statistical inference by applying theoretical knowledge to real-world datasets and scenarios. The project covers Monte Carlo simulation, confidence intervals, and hypothesis testing.

---

### Part 1: Roulette Simulation and Profit Analysis

#### Objective
To analyze the dynamics of a simple betting strategy in the game of American roulette using Monte Carlo simulation. The primary goal is to understand the distribution of total and average earnings over a varying number of rounds and to see the Central Limit Theorem in action.

#### Methodology
1.  **Game Simulation:** A Python function was developed to simulate playing `N` rounds of roulette, betting $1 on black each round. The function calculates the total earnings ($S_N$) after `N` rounds.
2.  **Monte Carlo Analysis:** The simulation was run 100,000 times for different values of `N` (10, 25, 100, 1000) to study the distribution of both total earnings ($S_N$) and average earnings ($S_N/N$).
3.  **Theoretical vs. Simulated:** The theoretical expected values and standard errors were calculated and compared against the results obtained from the simulation to validate the model.
4.  **Central Limit Theorem (CLT):** The CLT was used to approximate the probability of the casino losing money for `N=25` rounds, and this was verified using the simulation results. The analysis was extended to show how this probability changes as `N` increases.

#### Key Concepts Demonstrated
* Monte Carlo Simulation
* Probability Distributions
* Expected Value and Standard Error
* Central Limit Theorem (CLT)

---

### Part 2: Predicting the 2016 US Presidential Election

#### Objective
To analyze polling data from the 2016 U.S. presidential election between Donald Trump and Hillary Clinton. This part focuses on aggregating data from multiple polls to derive more precise estimates, calculate confidence intervals, and visualize trends over time.

#### Methodology
1.  **Data Cleaning:** Loaded the `2016-general-election-trump-vs-clinton.csv` dataset and filtered out irrelevant rows and columns.
2.  **Confidence Intervals:** Derived and computed the 95% confidence interval for the true proportion of voters supporting a candidate.
3.  **Time-Series Visualization:** Plotted the poll results over time for both candidates, including a smoothed trend line to better visualize shifts in support.
4.  **Data Aggregation:** Aggregated the results from all polls to calculate the overall estimated proportion of voters for each candidate and the corresponding 95% confidence intervals.
5.  **Hypothesis Testing:** Defined the "spread" ($d = p_{Clinton} - p_{Trump}$) and performed a hypothesis test to determine if the observed spread was statistically different from zero.

#### Key Concepts Demonstrated
* Confidence Intervals
* Data Aggregation
* Time-Series Analysis
* Hypothesis Testing (z-test)
* Standard Error of the Proportion

---

### Part 3: Drug Safety Trial Analysis

#### Objective
To determine if a new drug has a statistically significant effect on patients compared to a placebo. This was achieved by analyzing a dataset from a randomized controlled drug trial and performing hypothesis tests on key biological markers.

#### Methodology
1.  **Data Preparation:** Loaded the `drug_safety.csv` dataset and performed necessary cleaning, including handling missing values and transforming categorical columns for numerical analysis.
2.  **Exploratory Data Analysis (EDA):** Grouped the data by treatment type ('Drug' vs. 'Placebo') and calculated summary statistics for white blood cell count (WBC), red blood cell count (RBC), and the number of adverse effects.
3.  **Hypothesis Testing:** For each key metric (mean WBC, mean RBC, etc.), an independent two-sample t-test was performed.
    * **Null Hypothesis ($H_0$):** There is no significant difference in the mean of the metric between the Drug and Placebo groups.
    * **Alternative Hypothesis ($H_1$):** There is a significant difference.
4.  **Interpretation:** The resulting p-values were interpreted to either reject or fail to reject the null hypothesis for each test, based on a significance level ($\alpha$) of 0.05.

#### Key Concepts Demonstrated
* Hypothesis Testing Framework
* Independent Two-Sample T-test
* P-value and Significance Level ($\alpha$)
* Exploratory Data Analysis (EDA)

---

### Technologies Used
* **Language:** Python
* **Libraries:**
    * Pandas
    * NumPy
    * Matplotlib
    * Seaborn
    * SciPy
    * Tabulate
* **Environment:** Jupyter Notebook

### How to Run the Code

To set up and run this project on your local machine, please follow these steps. It is highly recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Amir-rfz/Data-Science-Course-Projects-S2025.git
    cd Data-Science-Course-Projects-S2025/CA0 - Statistical Inference
    ```

2.  **Create and activate a virtual environment:**
    * **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    All required packages are listed in the `requirements.txt` file. Install them with the following command:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter and Run:**
    Start Jupyter Notebook from your terminal:
    ```bash
    jupyter notebook
    ```
    Navigate to the project's `.ipynb` file, open it, and run the cells sequentially to see the analysis.

