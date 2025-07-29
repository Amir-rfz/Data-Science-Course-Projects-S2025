# Project 5 & 6: Advanced Topics in Data Science

## Project Overview

This repository contains the solutions for a final, multi-part assignment covering a range of advanced topics in modern data science. The project is designed to bridge the gap between traditional machine learning and the capabilities of foundation models, tackling challenges in natural language processing, computer vision, and low-resource learning environments.

The project is divided into four main tasks:

1.  **Semi-Supervised Learning:** Predicting video game review scores using a small labeled dataset and a large unlabeled dataset.

2.  **Semantic Search:** Building a semantic search engine for a Persian-language Q&A forum.

3.  **Large Language Models:** Using LLMs for complex multiple-choice reasoning on the SWAG dataset.

4.  **Unsupervised Image Segmentation:** Applying clustering methods to segment football players from images.

---

### Task 1: Semi-Supervised Learning for Video Game Reviews

#### Objective
To predict numerical review scores (1-10) for video game summaries in a low-resource setting. The primary challenge is to effectively leverage a large pool of unlabeled text data to improve the performance of a model trained on a very small set of labeled data.

#### Methodology

1.  **Text Vectorization:** Review summaries were converted into numerical embeddings using two methods:

    * **Sentence-Transformers:** Using the `all-MiniLM-L6-v2` model to generate dense, semantically rich sentence embeddings.

    * **Word2Vec:** Training a model on the entire text corpus and averaging word vectors to create sentence embeddings.

2.  **Supervised Baseline:** Baseline models (both classification and regression) were trained using only the small labeled dataset to establish initial performance benchmarks.

3.  **Pseudo-Labeling (SSL):** The best baseline model was used to predict scores for the unlabeled data. High-confidence predictions were used as "pseudo-labels" to augment the training set, and the model was retrained.

4.  **Active Learning:** An interactive learning loop was simulated. The model identified the most "uncertain" unlabeled samples, which were then "manually labeled" and added to the training set to iteratively improve the model with minimal labeling effort.

5.  **Comparative Analysis:** The performance of the baseline, pseudo-labeling, and active learning models was compared using metrics like F1-score, MAE, and learning curves to evaluate the effectiveness of each strategy.

---

### Task 2: Semantic Search Engine for Persian Q&A

#### Objective
To build a semantic search engine for the Persian-language NiniSite Q&A dataset. The goal was to move beyond simple keyword matching and retrieve answers that are semantically relevant to a user's query.

#### Methodology

1.  **Persian Text Preprocessing:** A comprehensive cleaning pipeline was applied to the informal Persian text, including character normalization, stopword removal, and lemmatization using the `hazm` library.

2.  **Embedding Model:** The multilingual `bge-m3` model was used to generate dense vector embeddings for all questions in the dataset, capturing their semantic meaning.

3.  **Vector Database:** `LanceDB` was used as a vector database to store the question text and their corresponding embeddings. LanceDB's ability to use a custom embedding function allowed for seamless integration.

4.  **Semantic & Full-Text Search:** The system was tested using both semantic (vector) search and traditional full-text search. The relevance of the results from both methods was manually evaluated and compared.

5.  **Answer Reranking (Bonus):** A `bge-reranker` model was applied to the top results from the semantic search to further refine the answer rankings based on a more precise (question, answer) pair evaluation.

---

### Task 3: LLMs for Commonsense Reasoning (SWAG)

#### Objective
To explore the capabilities of Large Language Models (LLMs) on a complex reasoning task that is challenging for traditional models. The SWAG dataset, a collection of multiple-choice questions about real-world scenarios, was used for this purpose.

#### Methodology

1.  **Baseline Evaluation:** A pre-trained BERT model (`bert-base-uncased`) was loaded and evaluated on the SWAG validation set in a zero-shot setting to establish a baseline.

2.  **In-Context Learning (ICL):** Few-shot learning was applied by formatting the input prompt to include several example question-answer pairs. The model's performance was evaluated to see if it could learn the task from context without any weight updates.

3.  **Fine-Tuning with LoRA:** The BERT model was fine-tuned on the SWAG training set using Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique. This adapts the model to the specific task while keeping most of its original weights frozen.

4.  **Combined Approach (Bonus):** The fine-tuned model was tested again using the same ICL prompt structure to see if combining fine-tuning with in-context examples provided any additional performance benefits.

5.  **Comprehensive Evaluation:** Performance across all approaches was measured using accuracy and perplexity, and the results were compared to analyze the effectiveness of each method.

---

### Task 4: Unsupervised Image Segmentation with Clustering

#### Objective
To perform semantic segmentation on images of football players using unsupervised clustering methods. The goal was to automatically generate segmentation masks that separate players from the background without relying on pre-labeled data.

#### Methodology

1.  **Feature Creation:** Images were downscaled to a manageable size. Pixel features were engineered for clustering. This included simple features like RGB color values and more complex ones combining color with spatial information (pixel coordinates).

2.  **Pixel Clustering:** Various clustering algorithms (K-Means, DBSCAN, Agglomerative Clustering) were applied to the pixel features. Hyperparameters (like the number of clusters `k` for K-Means) were tuned using metrics like the Silhouette score.

3.  **Filtering and Merging:** The resulting clusters were post-processed. Small, noisy clusters were filtered out, and adjacent clusters likely belonging to the same player were merged to form more coherent segments.

4.  **Binary Mask Generation:** The final clusters were used to create a binary segmentation mask, where pixels belonging to players were labeled `1` and background pixels were labeled `0`.

5.  **Evaluation:** The generated masks were compared against the ground-truth annotations provided. The quality of the segmentation was quantitatively measured using the **Intersection over Union (IoU)** and **Dice Coefficient** metrics.

---

### Technologies Used

* **Core Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

* **NLP & Embeddings:** Sentence-Transformers, Gensim, `hazm`, Hugging Face `transformers`, `datasets`

* **Vector Database:** LanceDB

* **Deep Learning:** PyTorch, TensorFlow

* **Computer Vision:** OpenCV (for image processing)

* **Utilities:** `tqdm`, `missingno`, `persian-reshaper`, `python-bidi`

### How to Run the Code

Each of the four tasks is self-contained in its own Jupyter Notebook.

1.  **Install Dependencies:**

    First, set up a Python environment (a virtual environment is recommended). Install all required packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Notebooks:**

    Navigate to the project folder and open the Jupyter Notebook for the task you wish to run (e.g., `Task_1_SSL.ipynb`, `Task_2_Semantic_Search.ipynb`, etc.).

3.  **Execute Cells:**

    Run the cells in each notebook sequentially. They are designed to handle all steps from data loading and preprocessing to model training and evaluation for that specific task.
