# Start the Dynamic Pricing GenAI Project

This file contains the exact commands to start the app stack in the correct order.

## 1) Open PowerShell in the project folder

```powershell
cd "C:\Users\Paras\Desktop\dynamic-pricing-genai\dynamic-pricing-genai"
```

## 2) Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

## 3) Start Ollama

Open a new PowerShell terminal and run:

```powershell
ollama serve
```

If you need to check the model list:

```powershell
ollama list
```

If the model is missing (the AI Assistant's primary model is `qwen2.5:7b`,
chosen for reliable tool-calling; `mistral` is only an automatic fallback if
`qwen2.5:7b` is unavailable):

```powershell
ollama pull qwen2.5:7b
ollama pull mistral
```

## 4) Start the FastAPI backend

Open another PowerShell terminal and run:

```powershell
cd "C:\Users\Paras\Desktop\dynamic-pricing-genai\dynamic-pricing-genai"
.\venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload
```

Check the API:

```powershell
http://127.0.0.1:8000/docs
```

## 5) Start the Streamlit app

Open another PowerShell terminal and run:

```powershell
cd "C:\Users\Paras\Desktop\dynamic-pricing-genai\dynamic-pricing-genai"
.\venv\Scripts\Activate.ps1
python -m streamlit run app\streamlit_app.py
```

Use `python -m streamlit run ...`, not the bare `streamlit run ...` --
the bare form fails with `ModuleNotFoundError: No module named 'app'` /
`'api'` because Streamlit's script runner only ever puts `app\` on
`sys.path`, never the project root that `app.route_service` and
`api.pricing_service` imports need. `python -m` adds the current
directory (the project root) to `sys.path` itself.

Open the app in the browser:

```text
http://localhost:8501
```

## 6) Optional: test the AI assistant manually

From a terminal:

```powershell
cd "C:\Users\Paras\Desktop\dynamic-pricing-genai\dynamic-pricing-genai"
.\venv\Scripts\Activate.ps1
python -c "import ollama; response = ollama.chat(model='qwen2.5:7b', messages=[{'role': 'user', 'content': 'What is surge pricing in ride-hailing? Answer in one sentence.'}], stream=False); print(response.get('message', {}).get('content', 'ERROR'))"
```

## 7) Optional: test the prediction API manually

```powershell
cd "C:\Users\Paras\Desktop\dynamic-pricing-genai\dynamic-pricing-genai"
.\venv\Scripts\Activate.ps1
python -c "import requests; payload={'distance': 5.0, 'surge_multiplier': 2.0, 'hour_of_day': 18, 'day_of_week': 4, 'is_weekend': False, 'is_rush_hour': True, 'is_raining': True, 'cab_type_encoded': 1, 'name_encoded': 1}; r = requests.post('http://127.0.0.1:8000/predict', json=payload, timeout=10); print(r.status_code); print(r.json())"
```

## 8) Full startup sequence in one block

If you want all project commands in one place:

```powershell
# Terminal 1
ollama serve

# Terminal 2
cd "C:\Users\Paras\Desktop\dynamic-pricing-genai\dynamic-pricing-genai"
.\venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload

# Terminal 3
cd "C:\Users\Paras\Desktop\dynamic-pricing-genai\dynamic-pricing-genai"
.\venv\Scripts\Activate.ps1
python -m streamlit run app\streamlit_app.py
```

## 9) Troubleshooting

### If Streamlit says Plotly missing

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install plotly==6.0.0
```

### If AI assistant cannot connect to Ollama

```powershell
ollama serve
ollama list
```

### If the API is not running

```powershell
cd "C:\Users\Paras\Desktop\dynamic-pricing-genai\dynamic-pricing-genai"
.\venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload
```

## 10) Project folders

- App UI: `app/`
- Backend API: `api/`
- Data: `data/`
- Trained models: `model/`
- Notebooks: `notebooks/`
- Requirements: `requirements.txt`
- Environment: `.env`

This is the main run guide for the project.
