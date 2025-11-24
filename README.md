# Chinese Online Content for LLM 🇨🇳

This repository contains curated Chinese online content designed for Large Language Model (LLM) training and evaluation. The content is organized as a Jekyll-based website for easy browsing and access.

## 📚 About

This project aims to collect and organize high-quality Chinese language content that can be used for:
- Training and fine-tuning LLMs on Chinese language data
- Evaluating LLM performance on Chinese language understanding
- Research and development of multilingual LLMs
- Educational purposes and linguistic studies

## 🚀 Jekyll on MacOS + GitHub Pages Deployment

This guide walks you through setting up your MacOS dev env to build and run Jekyll locally, and deploy your Jekyll site to GitHub Pages.

---

## ✅ Prerequisites

- macOS
- [Homebrew](https://brew.sh/)
- GitHub account
- Git installed and configured

---

## 🧰 Setup Instructions

### 1. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### 2. Install Ruby via rbenv
Ruby 3.3.x is not compatible with Jekyll 4.3.x. Use Ruby 3.2.2 instead. Follow the steps below
```bash
brew install rbenv ruby-build
echo 'eval "$(rbenv init -)"' >> ~/.zshrc
source ~/.zshrc
rbenv install 3.2.2
rbenv global 3.2.2
```

Verify installation
```bash
ruby -v  # should show ruby 3.2.2
```

### 3. Install Jekyll and Bundler
```bash
gem install bundler jekyll
```

### 4. Build and Run Locally
```bash
cd chinese-online-content-for-llm
bundle install
bundle exec jekyll serve
```
Visit http://localhost:4000 in your browser.

## 🚀 Deploying to GitHub Pages
### 1. Create a GitHub Repository
Name it: chinese-online-content-for-llm

### 2. Initialize Git and Push Site
```bash
cd chinese-online-content-for-llm
git init
git remote add origin https://github.com/yourusername/chinese-online-content-for-llm.git
git add .
git commit -m "Initial commit"
git push -u origin main
```
### 3. Configure _config.yml
In your Jekyll site directory:
```yaml
# _config.yml
url: "https://yourusername.github.io/chinese-online-content-for-llm"
baseurl: "/chinese-online-content-for-llm"
```
### 4. Enable GitHub Pages
Go to your repo on GitHub

Settings → Pages

Source: set to Deploy from a branch, use main or master, and / as the folder.

GitHub will serve your site at:
👉 https://yourusername.github.io/chinese-online-content-for-llm

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

See LICENSE.txt for details.
