# Project 2: Real-Time Payment Data Pipeline with Kafka, Spark, and MongoDB

## Project Overview

This project is a hands-on implementation of a scalable, real-time data pipeline designed to process financial transactions for a payment service provider. The system is engineered to ingest a high volume of transaction events, perform data validation, execute both batch and stream processing, detect fraud in real-time, and store the results for analysis and visualization.

The core technologies used are Apache Kafka for the data ingestion layer, Apache Spark (PySpark) for the processing layer (both batch and streaming), and MongoDB for the data storage layer.

---

## Pipeline Architecture

The data flows through the pipeline in the following sequence:

1.  **Transaction Generator:** A Python script simulates a non-homogeneous Poisson process to generate synthetic payment transaction data, mimicking real-world daily and weekly patterns.
   
2.  **Data Ingestion (Kafka):** The generated transactions are published as events to an Apache Kafka topic named `darooghe.transactions`.
   
3.  **Data Processing (Spark):** An Apache Spark application consumes the data from Kafka. It operates in two modes:
   
    * **Real-Time (Spark Streaming):** A streaming job processes incoming transactions in micro-batches to perform real-time fraud detection and calculate live commission metrics. Invalid data is routed to an `darooghe.error_logs` topic, and fraud alerts are sent to a `darooghe.fraud_alerts` topic.
      
    * **Batch Processing:** A separate batch job runs on historical data (or aggregates of streamed data) to perform deeper analysis, such as commission efficiency reports and customer segmentation.
      
4.  **Data Storage (MongoDB):** The processed data from the batch jobs and aggregated insights are loaded into a MongoDB database, partitioned for efficient querying.
   
5.  **Visualization:** Python visualization libraries are used to query the processed data from MongoDB and create dashboards that display key business insights.

---

## Component Breakdown

### 1. Data Ingestion Layer

* **Technology:** Apache Kafka
  
* **Topics:**
  
    * `darooghe.transactions`: The main topic for incoming raw transaction events.
      
    * `darooghe.error_logs`: For transactions that fail validation checks.
      
    * `darooghe.fraud_alerts`: For events flagged by the real-time fraud detection system.
      
* **Functionality:** A Kafka consumer reads from the `transactions` topic, deserializes the JSON events, and performs initial data validation based on business rules (e.g., amount consistency, timestamp validity, device info).

### 2. Batch Processing Layer

* **Technology:** PySpark
  
* **Functionality:**
  
    * **Commission Analysis:** Runs batch jobs on historical data to analyze commission efficiency by merchant category, comparing different commission structures (`flat`, `progressive`, `tiered`).
      
    * **Transaction Pattern Analysis:** Identifies temporal patterns, peak transaction hours, and customer segments based on historical spending habits.
      
    * **Data Storage:** Loads the cleaned and aggregated data into a MongoDB collection for long-term analysis and reporting.

### 3. Real-Time Processing Layer

* **Technology:** Spark Streaming
  
* **Functionality:**
  
    * **Micro-Batch Processing:** Consumes data from Kafka in time-based windows (e.g., 1-minute windows sliding every 20 seconds) for near real-time analysis.
      
    * **Fraud Detection:** Implements stateful streaming to detect fraudulent patterns based on three key rules:
      
        1.  **Velocity Check:** > 5 transactions from the same customer within 2 minutes.
           
        3.  **Geographical Impossibility:** Transactions from the same customer > 50 km apart within 5 minutes.
           
        5.  **Amount Anomaly:** A transaction amount that is >1000% of the customer's historical average.
           
    * **Real-Time Analytics:** Calculates live metrics such as total commission by type per minute and identifies top commission-generating merchants in 5-minute windows.

### 4. Visualization Layer

* **Technology:** Python (Matplotlib, Seaborn, Plotly)
  
* **Functionality:**
  
    * **Transaction Volume Dashboard:** Time-series charts showing both real-time and historical transaction volumes.
      
    * **Merchant Analysis:** Bar charts displaying the top 5 merchants by transaction count and total commission.
      
    * **User Activity Insights:** Visuals representing transactions per user and other engagement metrics.

---

### Technologies Used

* **Data Streaming:** Apache Kafka, ZooKeeper
  
* **Data Processing:** Apache Spark (PySpark), Spark Streaming
  
* **Database:** MongoDB
  
* **Monitoring:** Kafdrop (for viewing Kafka topics)

### How to Run the Pipeline

1.  **Environment Setup:** Ensure Java, Apache Kafka, Apache Spark, and MongoDB are installed and running.
   
2.  **Install Dependencies:**
   
    ```bash
    pip install -r requirements.txt
    ```
    
3.  **Start Services:**
   
    * Start ZooKeeper: `bin/zookeeper-server-start.sh config/zookeeper.properties`
      
    * Start Kafka Broker: `bin/kafka-server-start.sh config/server.properties`
      
    * Launch Kafdrop Monitoring UI: `java -jar kafdrop-4.1.0.jar --kafka.brokerConnect=localhost:9090` (Open your browser and go to `http://localhost:9000`)
      
    * Run the Transaction Generator: `python darooghe_pulse.py`
      
4.  **Run the Cells:** Run the cells sequentially to view the insights
