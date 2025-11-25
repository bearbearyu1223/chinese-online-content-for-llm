---
layout: page
---

<style>
.hero-banner {
  width: 100%;
  max-width: 100%;
  margin: 0 auto 2em auto;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

.hero-banner img {
  width: 100%;
  height: auto;
  display: block;
}

.welcome-section {
  text-align: center;
  margin: 2em 0;
  padding: 1.5em;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 8px;
}

.welcome-section h2 {
  margin-top: 0;
  color: #2c3e50;
}

.welcome-section p {
  font-size: 1.1em;
  color: #34495e;
  line-height: 1.6;
}

.collections-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 2em;
  margin: 3em 0;
}

.collection-card {
  border: 2px solid var(--border-color);
  border-radius: 12px;
  padding: 2em;
  text-align: center;
  transition: all 0.3s ease;
  background: var(--bg-color);
  box-shadow: 0 4px 12px var(--shadow);
}

.collection-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 8px 24px var(--shadow);
  border-color: var(--link-color);
}

.collection-card .icon {
  font-size: 4em;
  margin-bottom: 0.3em;
}

.collection-card h3 {
  margin: 0.5em 0;
  font-size: 1.5em;
}

.collection-card p {
  color: var(--text-color);
  opacity: 0.8;
  margin: 1em 0;
}

.collection-card .post-count {
  font-size: 0.9em;
  color: var(--link-color);
  font-weight: bold;
}

.collection-card a {
  text-decoration: none;
  color: inherit;
}

.recent-posts {
  margin-top: 4em;
}

.recent-posts h2 {
  text-align: center;
  margin-bottom: 2em;
}

.post-list {
  list-style: none;
  padding: 0;
}

.post-list li {
  margin-bottom: 1.5em;
  padding: 1.5em;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--bg-color);
}

.post-list .post-meta {
  font-size: 0.9em;
  color: #666;
  margin-bottom: 0.5em;
}

.post-list .post-link {
  font-size: 1.3em;
  font-weight: bold;
}

[data-theme="dark"] .welcome-section {
  background: linear-gradient(135deg, #2d333b 0%, #1a1f26 100%);
}

[data-theme="dark"] .welcome-section h2,
[data-theme="dark"] .welcome-section p {
  color: #e0e0e0;
}

[data-theme="dark"] .post-list .post-meta {
  color: #a0aec0;
}
</style>

<div class="hero-banner">
  <img src="{{ site.baseurl }}/assets/picture/cover-image.png" alt="🍒 大模型我都爱 - 中文 AI 技术学习站">
</div>

<div class="welcome-section">
  <h2>欢迎来到 🍒 大模型我都爱！</h2>
  <p>
    这里是一个专注于分享中文优质内容的平台<br>
    探索大语言模型的奥秘，分享 AI 技术学习笔记与职场成长经验
  </p>
</div>

{% assign published_cs336_count = site.data.published_posts.cs336 | size %}
{% assign published_career_count = site.data.published_posts.career | size %}
{% assign published_ai_count = site.data.published_posts.ai-insights | size %}

<h2 style="text-align: center; margin: 2em 0;">📚 内容专栏 Collections</h2>

<div class="collections-grid">
  <a href="{{ '/cs336/' | relative_url }}">
    <div class="collection-card">
      <div class="icon">📚</div>
      <h3>Stanford CS336</h3>
      <p>Language Modeling from Scratch<br>语言模型从零开始</p>
      <span class="post-count">{{ published_cs336_count }} posts</span>
    </div>
  </a>

  <a href="{{ '/career/' | relative_url }}">
    <div class="collection-card">
      <div class="icon">🌱</div>
      <h3>Career & Growth</h3>
      <p>Professional Development<br>职场成长与自我反思</p>
      <span class="post-count">{{ published_career_count }} posts</span>
    </div>
  </a>

  <a href="{{ '/ai-insights/' | relative_url }}">
    <div class="collection-card">
      <div class="icon">🧬</div>
      <h3>AI Insights</h3>
      <p>Latest AI Trends<br>人工智能趋势与洞察</p>
      <span class="post-count">{{ published_ai_count }} posts</span>
    </div>
  </a>
</div>

<div class="recent-posts">
  <h2>📝 最近更新 Recent Posts</h2>

  {% comment %}Filter CS336 posts{% endcomment %}
  {% assign cs336_published = "" | split: "" %}
  {% for post in site.cs336 %}
    {% assign filename = post.path | split: "/" | last %}
    {% if site.data.published_posts.cs336 contains filename %}
      {% assign cs336_published = cs336_published | push: post %}
    {% endif %}
  {% endfor %}

  {% comment %}Filter Career posts{% endcomment %}
  {% assign career_published = "" | split: "" %}
  {% for post in site.career %}
    {% assign filename = post.path | split: "/" | last %}
    {% if site.data.published_posts.career contains filename %}
      {% assign career_published = career_published | push: post %}
    {% endif %}
  {% endfor %}

  {% comment %}Filter AI Insights posts{% endcomment %}
  {% assign ai_published = "" | split: "" %}
  {% for post in site.ai-insights %}
    {% assign filename = post.path | split: "/" | last %}
    {% if site.data.published_posts.ai-insights contains filename %}
      {% assign ai_published = ai_published | push: post %}
    {% endif %}
  {% endfor %}

  <ul class="post-list">
    {% assign all_posts = cs336_published | concat: career_published | concat: ai_published | sort: 'date' | reverse %}
    {% for post in all_posts limit:5 %}
      <li>
        <div class="post-meta">
          <span>{{ post.date | date: "%Y-%m-%d" }}</span>
          <span> • {{ post.collection_name }}</span>
        </div>
        <h3>
          <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title }}</a>
        </h3>
      </li>
    {% endfor %}
  </ul>
</div>
