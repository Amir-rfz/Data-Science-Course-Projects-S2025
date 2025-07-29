# Project 1: Langevin Dynamics Sampling & Tableau Data Storytelling

## Project Overview

This project is divided into two parts, each tackling a different area of data science. The first part focuses on the implementation of an advanced statistical sampling method, Langevin Dynamics. The second part applies data visualization and storytelling principles to create an interactive dashboard suite in Tableau using real-world Airbnb data.

---

### Part 1: Sampling with Langevin Dynamics

#### Objective
The primary goal of this section was to implement the Langevin Dynamics algorithm, a gradient-based Markov Chain Monte Carlo (MCMC) method, to sample from a probability distribution. This exercise provides a hands-on understanding of how modern sampling algorithms work, especially in scenarios where direct sampling is difficult.

#### Methodology
1.  **Target Distribution:** A 2D Gaussian distribution with a mean of `[-5, 5]` and a covariance of `5I` was defined as the target for our sampling algorithm.
2.  **Score Function Implementation:** The score function, defined as the gradient of the log-probability density function ($\nabla_{x} \log p(x)$), was theoretically derived and implemented in Python. This function is crucial as it guides the samples toward high-probability regions.
3.  **Score Field Visualization:** To verify the correctness of the score function, it was visualized as a quiver plot overlaid on the heatmap of the target distribution. The vectors correctly pointed towards the distribution's mean, indicating a correct implementation.
4.  **Langevin Dynamics Algorithm:** The iterative Langevin Dynamics algorithm was implemented. Starting from random points, the algorithm updates their positions using the score function and adds Gaussian noise at each step to ensure proper exploration of the sample space.
5.  **Trajectory Visualization:** The paths of the sampling points were plotted to visualize how they converge from their initial random positions to the high-density region of the target distribution.
6.  **Comparative Analysis:** 1,000 samples were generated using the implemented Langevin sampler and compared against 1,000 samples drawn using NumPy's standard `multivariate_normal` function. Visual inspection confirmed that both methods produced a similar distribution of points.

#### Key Concepts Demonstrated
* Markov Chain Monte Carlo (MCMC) Methods
* Langevin Dynamics
* Score Function (Gradient of Log-Probability)
* 2D Gaussian Distributions
* Procedural Algorithm Implementation in Python

---

### Part 2: Airbnb Data Storytelling with Tableau

#### Objective
To analyze a dataset of Airbnb listings in a specific city and present the findings as a cohesive, interactive data story using Tableau. The goal was to move beyond a single dashboard and create a multi-faceted narrative that allows users to explore the data dynamically.

#### Methodology
1.  **Data Integration and Preparation:** Two datasets, `Airbnb_Listings.xls` and `Neighborhood_Locations.xlsx`, were imported into Tableau. They were joined using the `neighborhood` field to enrich the listings data with geographical coordinates.
2.  **Dashboard Story Creation:** A story was constructed using three distinct dashboards:
    * **Dashboard 1: Market Overview:** This dashboard provided a high-level view of the Airbnb market, including KPIs for the total number of listings, average price, and number of hosts. A map showed the geographical distribution of listings across different neighborhood groups.
    * **Dashboard 2: Price and Availability Analysis:** This dashboard allowed for a deeper dive into pricing and availability. It included visualizations showing the relationship between room type and price, and how availability changes across different neighborhoods. Interactive filters for price range and room type were included.
    * **Dashboard 3: Host and Review Insights:** The final dashboard focused on host activity and listing popularity. It visualized the number of listings per host and analyzed the correlation between the number of reviews and price.
3.  **Interactivity:** Filters, parameters, and dashboard actions were used throughout the story to enable users to dynamically explore the data. For example, clicking a neighborhood on the map in the first dashboard would filter the data in the subsequent dashboards.
4.  **Design Principles:** Gestalt principles and preattentive attributes (color, size) were used to create a clean, intuitive, and easy-to-read set of dashboards that effectively guided the user through the data narrative.
5.  **Publication:** The final story was published to Tableau Public for accessibility.

#### Key Concepts Demonstrated
* Data Storytelling
* Interactive Dashboard Design
* Key Performance Indicators (KPIs)
* Geospatial Analysis (Mapping)
* Data Blending/Joining
* Gestalt Principles in Visualization

---

### Technologies Used
* **Part 1 (Sampling):**
    * **Language:** Python
    * **Libraries:** NumPy, Matplotlib, SciPy
    * **Environment:** Jupyter Notebook
* **Part 2 (Visualization):**
    * **Software:** Tableau Desktop
    * **Platform:** Tableau Public

### How to Run or View This Project

#### Part 1: Langevin Dynamics Code
The Python code for the sampling implementation is contained in a Jupyter Notebook.
1.  All required packages are listed in the requirements.txt file. Install them with the following command:
    ```bash
    pip install -r requirements.txt
    ```
2.  Launch Jupyter Notebook and open the `.ipynb` file.
3.  Run the cells sequentially to reproduce the visualizations and analysis.

#### Part 2: Tableau Story
you can see the interactive data story by opening .twb file in tableau .
* ** Also you can view the report of this Task here: [report](https://github.com/Amir-rfz/Data-Science-Course-Projects-S2025/blob/main/CA1%20-%20Sampling%20and%20Data%20Storytelling/Task2/task2-report.pdf)**
