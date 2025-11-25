---
layout: page
title: 🧬 AI Insights
permalink: /ai-insights/
---

<style>
.collection-header {
  text-align: center;
  padding: 2em 0;
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
  border-radius: 12px;
  margin-bottom: 2em;
}

.collection-header h1 {
  margin: 0;
  font-size: 2.5em;
}

.collection-header p {
  font-size: 1.2em;
  margin: 0.5em 0 0 0;
  opacity: 0.9;
}

.posts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5em;
  margin: 2em 0;
}

.post-card {
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 1.5em;
  background-color: var(--bg-color);
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px var(--shadow);
}

.post-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 16px var(--shadow);
}

.post-card h3 {
  margin-top: 0;
  font-size: 1.3em;
}

.post-card .post-date {
  color: #666;
  font-size: 0.9em;
  margin-bottom: 0.5em;
}

[data-theme="dark"] .collection-header {
  background: linear-gradient(135deg, #833ab4 0%, #fd1d1d 100%);
}

[data-theme="dark"] .post-card .post-date {
  color: #a0aec0;
}
</style>

{% assign published_ai = site.data.published_posts.ai-insights | size %}

<div class="collection-header">
  <h1>🧬 AI Insights</h1>
  <p>Latest AI Trends & Analysis | 人工智能趋势与洞察</p>
  <p style="font-size: 0.9em; margin-top: 1em;">{{ published_ai }} posts</p>
</div>

## 关于这个系列 About This Series

探索人工智能领域的最新动态、前沿技术和深度分析。分享对AI发展趋势的观察与思考。

Exploring the latest developments, cutting-edge technologies, and in-depth analysis in the field of artificial intelligence. Sharing observations and reflections on AI trends.

---

## 📝 所有文章 All Posts

{% assign published_list = site.data.published_posts.ai-insights %}
{% assign published_posts = "" | split: "" %}
{% for post in site.ai-insights %}
  {% assign filename = post.path | split: "/" | last %}
  {% if published_list contains filename %}
    {% assign published_posts = published_posts | push: post %}
  {% endif %}
{% endfor %}

<div class="posts-grid">
{% assign sorted_posts = published_posts | sort: 'date' | reverse %}
{% if sorted_posts.size > 0 %}
  {% for post in sorted_posts %}
    <div class="post-card">
      <div class="post-date">{{ post.date | date: "%Y-%m-%d" }}</div>
      <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
    </div>
  {% endfor %}
{% else %}
  <p style="text-align: center; color: #666; font-style: italic;">Coming soon... 即将推出...</p>
{% endif %}
</div>
