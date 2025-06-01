## [Paper Review] Open FinLLM Leaderboard: Towards Financial AI Readiness

- **Paper Link**
  - [Link](https://arxiv.org/pdf/2501.10963)
  - Date: 25.01
- **Key Takeaway**
  - FinGPT Search Agents
    - Retrieve trustful information from resources using FinGPT based Agent (RAG)
        - Resource: Yahoo, Finance, Bloomberg, local documents like PDFs and Excel sheets
    - Reference:
        - [Github](https://github.com/Open-Finance-Lab/FinGPT-Search-Agent)
  - Open FinLLM Leaderboard
    - Evaluation Framework: Provide transparent and standard evaluation framework
    - Zero-shot Evaluation: w/o Fine-tuning on the task-specific dataset
    - Model download: API or Huggingface
    - Evaluation Metric: Appropriate metrics according to task + Normalization through Min-Max Scaling
        - Task: Financial Reporting, Sentiment Analysis, Stock Prediction, etc.
        - Multimodal: Text, Table, Numerical data, and structured format like XBRL
    - [task overview](./images/task_overview.png)
    - [datasets link](./images/dataset_link.png)
    - Reference:
      - [Github](https://github.com/finos-labs/Open-Financial-LLMs-Leaderboard)
      - [Huggingface Demo](https://huggingface.co/spaces/finosfoundation/Open-Financial-LLM-Leaderboard)
      - [Evaluation examples notebook](https://github.com/finos-labs/Open-Financial-LLMs-Leaderboard/blob/main/examples/model_evaluation.ipynb)
      - [PIXIU Project Github](https://github.com/The-FinAI/PIXIU?tab=readme-ov-file)
  - Related Works
    - FinLLM
      - BloombergGPT, FinGPT
    - Benchmark
      - FinBen (2402)
        - [paper link](https://arxiv.org/pdf/2402.12659)
        - 24 financial datasets & 36 datasets
      - FinanceBench (2311)
        - [paper link](https://arxiv.org/pdf/2311.11944)

## [Paper Review] A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges