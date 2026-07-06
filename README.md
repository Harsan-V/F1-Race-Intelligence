# F1 Race Intelligence

F1 Race Intelligence is an end-to-end Formula 1 analytics project that combines machine learning, race data analysis, explainable AI, and Generative AI. The project includes modules for race winner prediction, lap time prediction, driver performance classification, pit stop strategy recommendation, an F1 chatbot using RAG, race summary generation, and SHAP-based model explanation.

## Project Overview

Formula 1 decisions depend on many variables: driver pace, qualifying performance, tire behavior, lap history, race conditions, team performance, and strategy. This project explores those signals through multiple machine learning and AI modules built around F1 race intelligence.

The goal is to demonstrate how data-driven systems can support race analysis, performance prediction, strategy recommendations, and natural-language insight generation.

## Features

- Predict possible Formula 1 race winners using historical and session-based data.
- Predict lap times using driver, team, circuit, session, and recent pace features.
- Classify driver performance using race and session metrics.
- Recommend pit stop strategy windows and estimate finishing positions.
- Build an F1 chatbot using Retrieval-Augmented Generation with FAISS, LangChain, and Groq.
- Generate race summaries with LLM-based text generation.
- Explain model predictions using SHAP visualizations.
- Provide FastAPI backend endpoints for selected modules.
- Include Jupyter notebooks for experimentation, learning, and demonstration.

## Tech Stack

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastF1
- SHAP
- Matplotlib
- Seaborn
- FastAPI
- Uvicorn
- LangChain
- FAISS
- Sentence Transformers
- Groq API

## Repository Structure

```text
F1-Race-Intelligence/
├── backend/                          # FastAPI backend for project modules
├── data/                             # Input datasets
├── artifacts/                        # Generated artifacts and model outputs
├── module_2_lap_time_prediction/     # Lap time prediction scripts and artifacts
├── module_2_lap_time_prediction_v2/  # Alternate lap time prediction version
├── module_4_pit_stop_strategy/       # Pit stop strategy recommendation module
├── module_5_f1_chatbot/              # GenAI + RAG chatbot module
├── session_model_data/               # Session-level model data
├── Module_1_*.ipynb                  # Race winner prediction notebooks
├── Module_2_*.ipynb                  # Lap time prediction notebooks
├── Module_3_*.ipynb                  # Driver performance classification notebook
├── Module_4_*.ipynb                  # Pit stop strategy recommendation notebook
├── Module_5_*.ipynb                  # F1 chatbot and RAG notebooks
├── Module_6_*.ipynb                  # Race summary generator notebook
├── Module_7_*.ipynb                  # Explainable AI with SHAP notebook
├── requirements.txt                  # Main Python dependencies
└── README.md                         # Project documentation
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Harsan-V/F1-Race-Intelligence.git
cd F1-Race-Intelligence
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Some modules use the Groq API for LLM-based chatbot and race summary generation.

Create a local `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Do not commit your real API key to GitHub. Keep real secrets only in `.env`.

## How to Run

### Run Jupyter Notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Recommended notebook order:

1. `Module_1_F1_Race_Winner_Prediction.ipynb`
2. `Module_2_F1_Lap_Time_Prediction.ipynb`
3. `Module_3_F1_Driver_Performance_Classification.ipynb`
4. `Module_4_F1_Pit_Stop_Strategy_Recommendation.ipynb`
5. `Module_5_F1_Chatbot_GenAI_RAG.ipynb`
6. `Module_6_F1_Race_Summary_Generator.ipynb`
7. `Module_7_F1_Explainable_AI_SHAP.ipynb`

### Run the FastAPI Backend

From the project root:

```bash
uvicorn backend.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

The `/docs` page provides an interactive Swagger UI for testing API endpoints.

## Modules

### Module 1: Race Winner Prediction

This module explores Formula 1 race winner prediction using historical race data, session results, practice performance, sprint data, and qualifying-related features. Multiple notebooks are included to show beginner, improved, and session-based approaches.

### Module 2: Lap Time Prediction

This module predicts representative lap time using circuit, session, driver, team, lap number, elapsed session time, previous-lap pace, rolling pace, and speed trap features.

Run the script-based pipeline:

```bash
python3 module_2_lap_time_prediction/build_dataset.py
python3 module_2_lap_time_prediction/train.py
python3 module_2_lap_time_prediction/predict.py --input module_2_lap_time_prediction/example_request.json
```

The model uses a chronological holdout so that evaluation better reflects future-race prediction.

### Module 3: Driver Performance Classification

This module classifies driver performance based on race/session features. It is designed to support performance grouping, comparison, and model-based driver analysis.

### Module 4: Pit Stop Strategy Recommendation

This module recommends pit stop windows and predicts finishing position using race results, lap-time history, estimated tire degradation trends, grid position, driver strength, team strength, and race identity.

Run the default recommendation:

```bash
python3 module_4_pit_stop_strategy/pit_strategy.py
```

Run a specific race:

```bash
python3 module_4_pit_stop_strategy/pit_strategy.py --season 2025 --round 4
```

### Module 5: F1 Chatbot with GenAI and RAG

This module answers Formula 1 questions using Retrieval-Augmented Generation.

Pipeline:

```text
Question -> Embeddings -> FAISS Vector DB -> Groq LLM -> Answer
```

Build the vector database:

```bash
python3 module_5_f1_chatbot/build_vector_db.py
```

Ask one question:

```bash
python3 module_5_f1_chatbot/f1_chatbot.py --question "Explain undercut strategy in Formula 1"
```

Run the interactive chatbot:

```bash
python3 module_5_f1_chatbot/f1_chatbot.py
```

### Module 6: Race Summary Generator

This module uses Groq LLM integration to generate structured race summaries and race insights from available race information.

### Module 7: Explainable AI with SHAP

This module uses SHAP to explain machine learning model predictions and show how individual features influence model output.

## Outputs
The F1 dashboard :
<img width="1470" height="956" alt="Screenshot 2026-07-06 at 11 50 59 AM" src="https://github.com/user-attachments/assets/8a5d66d7-c7fb-4e57-9b8f-538d9e4d23c9" />

<img width="1470" height="956" alt="Screenshot 2026-07-06 at 11 51 07 AM" src="https://github.com/user-attachments/assets/b391e343-9975-4e3d-8569-77da5eae97fd" />

<img width="1470" height="956" alt="Screenshot 2026-07-06 at 11 51 15 AM" src="https://github.com/user-attachments/assets/9193e308-dbac-4b5e-9a1c-0a2fd29908e3" />

<img width="1470" height="956" alt="Screenshot 2026-07-06 at 11 51 24 AM" src="https://github.com/user-attachments/assets/34b86781-fbba-4ff8-a0a0-320c74a0c6d2" />

<img width="1470" height="956" alt="Screenshot 2026-07-06 at 11 51 30 AM" src="https://github.com/user-attachments/assets/75ddf18d-1472-40a8-b46c-0f284cefa6b4" />

<img width="1470" height="956" alt="Screenshot 2026-07-06 at 11 51 35 AM" src="https://github.com/user-attachments/assets/527d12c3-c9f3-4db9-9115-b65c3ff29ffb" />

<img width="1470" height="956" alt="Screenshot 2026-07-06 at 11 52 32 AM" src="https://github.com/user-attachments/assets/ba504294-6060-43b2-9aa4-1aa95b914554" />

## Future Improvements

- Add a Streamlit or React dashboard for visual interaction.
- Improve model accuracy with more seasons and richer race features.
- Add live race/session data integration.
- Add automated model evaluation reports.
- Improve chatbot document retrieval quality.
- Add tests for backend endpoints and model utility functions.


## License

This project is developed for academic and educational purposes.
