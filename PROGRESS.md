Project Progress — Dynamic Pricing Engine (GenAI Edition)

Tracking checklist for build phases. Check boxes as you complete each item — GitHub renders - [x] as a clickable checkbox on the repo page.

Phase 0 — Setup ✅
 GitHub repo created
 Cloned into VS Code
 Virtual environment created + activated
 Packages installed, requirements.txt generated and pushed
 Ollama installed + qwen2.5:7b (or similar tool-calling model) pulled


 
Phase 1 — Foundation (Data + Features)
 Download dataset from Kaggle (Uber/Lyft, Airbnb, or flight prices)
 Load into Pandas, inspect, handle missing values
 Encode categorical fields (City, Car Type, etc.)
 Engineer time-based features (Is_Weekend, Is_Holiday, Hour_of_Day)
 Save cleaned dataset to data/


Phase 2 — Core Predictive Model
 Confounding sanity check — stratified price-vs-demand plots (save the comparison image)
 Train/test split
 Train XGBoost regressor, evaluate MAE/R²
 Add quantile regression or mapie for confidence intervals
 Document results in notebook


Phase 3 — Explainability & Serving
 Fit SHAP TreeExplainer
 Extract top-feature explanations per prediction
 Export model + explainer + uncertainty wrapper via joblib
 Build FastAPI /predict endpoint returning price + CI + SHAP features
 Stub /whatif endpoint


Phase 4 — GenAI Layer (Ollama)
 Build what_if_price_change() simulation function
 Wire it into /whatif
 Connect Ollama (qwen2.5:7b) for SHAP-grounded price justification
 Give the model the what_if_price_change tool via Ollama's tool-calling
 Test tool-calling accuracy across varied phrasings


Phase 5 — Frontend, Polish & Deployment
 ✅ Build Streamlit dashboard (prediction form + chat)
   ✅ 1_Price_Prediction.py — Cab price predictions
   ✅ 2_What_If_Simulator.py — Price reasonableness check
   ✅ 3_AI_Assistant.py — Chat interface with Ollama (tested & working)
   ✅ 4_SHAP_Explanations.py — Feature importance and interpretability
   ✅ 5_Analytics_Dashboard.py — Historical predictions and analytics
 ✅ Connect dashboard to FastAPI backend via requests
 ✅ Supabase database logging for all predictions
 📋 Write final README (architecture, dataset, confounding finding, how to run)
 📋 (Optional) Deploy demo

Phase 6 — Advanced Features (Complete)
 ✅ SHAP integration for explainability
 ✅ Dashboard with historical analysis
 ✅ Analytics filtering and performance tracking
 ✅ Multiple LLM support in AI Assistant