# 🚀 GitHub Repository Setup Guide

## Step-by-Step Instructions for MAHBUB

### 📋 Prerequisites Checklist
- [ ] Git installed on your computer
- [ ] GitHub account logged in
- [ ] Repository created: `Thai-AccidentIQ-AI`

---

## 🔧 Initial Setup (One-Time)

### 1. Configure Git (if not done)
```bash
git config --global user.name "mahbubchula"
git config --global user.email "6870376421@student.chula.ac.th"
```

### 2. Navigate to Your Project
```bash
cd "E:\ML Research\Thai accident data"
```

---

## 📦 Prepare Files for GitHub

### 3. Create .env File (LOCAL ONLY - Never commit!)
```bash
# Create .env file with your API key
echo GROQ_API_KEY=your_actual_groq_api_key_here > .env
```

### 4. Copy GitHub Files
Download these files and place in your project root:
- `README.md`
- `.gitignore`
- `LICENSE`
- `.env.example`
- `SETUP_INSTRUCTIONS.md` (this file)

### 5. Create .gitkeep Files
```bash
# Create placeholder files to keep empty directories in git
echo "" > data/raw/.gitkeep
echo "" > data/processed/.gitkeep
echo "" > models/.gitkeep
echo "" > outputs/figures/.gitkeep
echo "" > outputs/reports/.gitkeep
echo "" > outputs/results/.gitkeep
```

---

## 🚀 Push to GitHub

### 6. Initialize Git Repository
```bash
git init
```

### 7. Add All Files
```bash
git add .
```

### 8. Check What Will Be Committed (IMPORTANT!)
```bash
git status
```

**Verify:**
- ✅ .env is NOT in the list (should be ignored)
- ✅ .env.example IS in the list
- ✅ README.md, LICENSE, .gitignore are in the list

### 9. First Commit
```bash
git commit -m "Initial commit: Thai AccidentIQ AI - Complete ML pipeline with XAI and LLM"
```

### 10. Set Main Branch
```bash
git branch -M main
```

### 11. Add Remote Repository
```bash
git remote add origin https://github.com/mahbubchula/Thai-AccidentIQ-AI.git
```

### 12. Push to GitHub
```bash
git push -u origin main
```

---

## ✅ Verification

### Check Your Repository
1. Go to: https://github.com/mahbubchula/Thai-AccidentIQ-AI
2. Verify:
   - ✅ README.md is displayed
   - ✅ All files are there
   - ❌ .env is NOT there (security!)
   - ✅ .env.example IS there

---

## 🔐 Security Checklist

### CRITICAL: Verify No Secrets Committed!
```bash
# Search for API key in git history
git log --all --full-history --source -- **/*.py | grep -i "gsk_"
```

If you see your API key:
```bash
# Remove from history (dangerous - use carefully)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/file" \
  --prune-empty --tag-name-filter cat -- --all
```

---

## 📝 Future Updates

### Making Changes
```bash
# 1. Make your changes to files

# 2. Check what changed
git status

# 3. Add changes
git add .

# 4. Commit with message
git commit -m "Description of changes"

# 5. Push to GitHub
git push
```

---

## 🎯 Common Commands

```bash
# Check status
git status

# View changes
git diff

# View commit history
git log --oneline

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main

# Pull latest changes
git pull

# Clone repository (on another computer)
git clone https://github.com/mahbubchula/Thai-AccidentIQ-AI.git
```

---

## ⚠️ Important Notes

### Never Commit:
- ❌ `.env` file (contains API keys)
- ❌ Large data files (use Git LFS or exclude)
- ❌ Model files over 100MB
- ❌ Personal credentials

### Always Commit:
- ✅ `.env.example` (template without real keys)
- ✅ README.md
- ✅ LICENSE
- ✅ .gitignore
- ✅ Source code
- ✅ Requirements.txt

---

## 🆘 Troubleshooting

### Problem: "Permission denied"
```bash
# Use HTTPS instead of SSH
git remote set-url origin https://github.com/mahbubchula/Thai-AccidentIQ-AI.git
```

### Problem: "Files too large"
```bash
# Remove large files from staging
git reset HEAD path/to/large/file
```

### Problem: "Merge conflicts"
```bash
# Pull latest changes first
git pull origin main

# Resolve conflicts manually
# Then commit
git add .
git commit -m "Resolved conflicts"
git push
```

---

## 🎓 Best Practices

1. **Commit Often**: Small, frequent commits are better
2. **Meaningful Messages**: Describe what and why
3. **Check Before Push**: Always `git status` first
4. **Never Commit Secrets**: Use environment variables
5. **Document Changes**: Update README when adding features

---

## ✨ Done!

Your Thai AccidentIQ AI is now on GitHub! 🎉

**Repository**: https://github.com/mahbubchula/Thai-AccidentIQ-AI

**Next Steps:**
- Add topics/tags to repository
- Enable GitHub Pages (optional)
- Add collaborators (optional)
- Set up GitHub Actions (optional)

---

**Need Help?** Contact GitHub Support or check: https://docs.github.com
