# 🚀 Streamlit Cloud Deployment Guide

## Quick Deployment Steps

### 1. Sign Up for Streamlit Cloud
- Go to: https://share.streamlit.io/
- Sign in with GitHub
- Authorize Streamlit

### 2. Create New App
- Click "New app"
- Repository: `mahbubchula/Thai-AccidentIQ-AI`
- Branch: `main`
- Main file: `thai_accidentiq_ai.py`

### 3. Add Secret (IMPORTANT!)
In Advanced Settings > Secrets, add:
```toml
GROQ_API_KEY = "paste_your_actual_groq_api_key_here"
```

### 4. Deploy!
- Click "Deploy!"
- Wait 3-5 minutes
- Your app will be live!

### 5. Get Your URL
You'll receive a URL like:
```
https://thai-accidentiq-ai-xxxxx.streamlit.app
```

---

## 🔄 Update App
After making changes:
```bash
git push
```
App auto-redeploys in 1-2 minutes!

---

## 📞 Support
- Docs: https://docs.streamlit.io/streamlit-cloud
- Forum: https://discuss.streamlit.io/