# Bilingual Post Translation and Splitting Guide

This document provides step-by-step instructions for transforming English-only blog posts into bilingual (English-Chinese) format and splitting long posts into multiple parts.

## Overview

The bilingual format presents content in parallel structure: each English section is immediately followed by its Chinese translation. This approach maintains content integrity while making it accessible to both English and Chinese readers.

## Bilingual Format Structure

### 1. Front Matter

Update the post's YAML front matter:

```yaml
---
layout: post
title: "Your Post Title (English)"
date: YYYY-MM-DD
categories: [category-name]
author: 大模型我都爱
---
```

**Key changes:**
- Set `author: 大模型我都爱` (required for all bilingual posts)
- Keep the English title in the front matter
- Maintain original date and categories

### 2. Title Structure

After front matter, include both language versions of the title:

```markdown
# English Title

# 中文标题
```

### 3. Xiaohongshu Author Box

Add the Xiaohongshu author box immediately after titles (for posts intended for Xiaohongshu):

```html
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; color: white; margin: 20px 0; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);">
  <div style="display: flex; align-items: center; margin-bottom: 12px;">
    <span style="font-size: 24px; margin-right: 10px;">✨</span>
    <strong style="font-size: 18px;">关注我的小红书账号</strong>
  </div>
  <p style="margin: 8px 0; opacity: 0.95;">
    📱 <strong>小红书ID:</strong> AI_Builder_Greg
  </p>
  <p style="margin: 8px 0; font-size: 14px; opacity: 0.9;">
    🎯 分享AI学习笔记、技术教程、项目实战<br>
    💡 从入门到进阶，一起探索AI的无限可能
  </p>
  <p style="margin: 12px 0 0 0; font-size: 13px; opacity: 0.85;">
    👆 搜索关注，获取更多优质AI内容和学习资源
  </p>
</div>
```

### 4. Parallel Section Structure

For every content section, follow this pattern:

```markdown
## English Section Heading

English content paragraph...

## 中文章节标题

中文内容段落...

### English Subsection

English subsection content...

### 中文子章节

中文子章节内容...
```

**Important rules:**
- Each English heading is immediately followed by its Chinese equivalent
- Content paragraphs follow the same parallel structure
- Preserve ALL markdown formatting (code blocks, lists, tables, links, etc.)
- Keep the same heading hierarchy (# → ##, ### → ###)

### 5. Code Blocks and Examples

Code blocks remain in their original form (usually English) but add Chinese explanations:

```markdown
Here's an example implementation:

以下是示例实现：

```python
def example_function():
    return "Hello"
```
```

### 6. Lists

Convert numbered lists to bullet points for better readability:

**Before (numbered):**
```markdown
1. Token 73: 'I' (1 bytes)
2. Token 499: ' have' (5 bytes)
```

**After (bullets):**
```markdown
- Token 73: 'I' (1 bytes)
- Token 499: ' have' (5 bytes)
```

Apply this to both English and Chinese versions.

## Splitting Long Posts into Multiple Parts

### When to Split

Consider splitting a post when:
- Total length exceeds 800-1000 lines
- Content has natural logical divisions
- Multiple major topics are covered
- Reading time would exceed 15-20 minutes

### Splitting Strategy

1. **Identify Natural Break Points**
   - Look for major topic transitions
   - Find sections that are relatively self-contained
   - Aim for roughly equal part lengths

2. **Common Split Patterns**
   - Part 1: Theory/Background + Implementation
   - Part 2: Testing + Results + Examples

   OR

   - Part 1: Setup + Core Concepts
   - Part 2: Advanced Features + Examples

### Creating Part Files

**Naming Convention:**
```
YYYY-MM-DD-original-title-part1.md
YYYY-MM-DD-original-title-part2.md
```

Example:
```
2025-07-26-cs336-note-train-bpe-tinystories-part1.md
2025-07-26-cs336-note-train-bpe-tinystories-part2.md
```

**Front Matter for Parts:**

Part 1:
```yaml
---
layout: post
title: "Original Title: Topic Focus (Part 1)"
date: YYYY-MM-DD
categories: [category-name]
author: 大模型我都爱
---
```

Part 2:
```yaml
---
layout: post
title: "Original Title: Topic Focus (Part 2)"
date: YYYY-MM-DD
categories: [category-name]
author: 大模型我都爱
---
```

### Cross-Linking Between Parts

**At the end of Part 1:**
```markdown
---

Continue reading in [Part 2](/chinese-online-content-for-llm/cs336/original-title-part2/) for [brief description of Part 2 content].

继续阅读[第2部分](/chinese-online-content-for-llm/cs336/original-title-part2/)以查看[Part 2内容简述]。
```

**At the beginning of Part 2:**
```markdown
# Title (Part 2)

# 标题（第2部分）

This is Part 2 of [topic]. See [Part 1](/chinese-online-content-for-llm/cs336/original-title-part1/) for [brief description of Part 1 content].

这是[主题]的第2部分。查看[第1部分](/chinese-online-content-for-llm/cs336/original-title-part1/)了解[Part 1内容简述]。
```

**Link Format Rules:**
- Use relative URLs starting with `/chinese-online-content-for-llm/`
- Follow Jekyll permalink structure: `/chinese-online-content-for-llm/COLLECTION/POST-TITLE/`
- Post title in URL is derived from filename without date prefix
- Always end with trailing slash `/`

Examples:
```markdown
Correct: /chinese-online-content-for-llm/cs336/cs336-note-train-bpe-tinystories-part1/
Incorrect: https://bearbearyu1223.github.io/cs336/2025/07/26/cs336-note-train-bpe-tinystories-part1.html
```

## Complete Workflow Example

### Step 1: Analyze the Original Post

Original file: `_cs336/2025-07-26-cs336-note-train-bpe-tinystories.md`

- Check total length: 1200+ lines
- Identify content sections
- Determine split point: After "BPE Training Implementation"

### Step 2: Create Part 1

1. Copy original file to `2025-07-26-cs336-note-train-bpe-tinystories-part1.md`
2. Update front matter:
   ```yaml
   title: "Building a BPE Tokenizer from Scratch: Implementation (Part 1)"
   author: 大模型我都爱
   ```
3. Add bilingual title structure
4. Add Xiaohongshu author box
5. Transform content section by section:
   - English heading → Chinese heading
   - English content → Chinese translation
   - Maintain all code blocks, lists, formatting
6. Remove Part 2 content
7. Add cross-link to Part 2 at the end

### Step 3: Create Part 2

1. Copy original file to `2025-07-26-cs336-note-train-bpe-tinystories-part2.md`
2. Update front matter:
   ```yaml
   title: "Building a BPE Tokenizer from Scratch: Training Results and Testing (Part 2)"
   author: 大模型我都爱
   ```
3. Add bilingual title structure
4. Add reference to Part 1 at the beginning
5. Add Xiaohongshu author box
6. Remove Part 1 content, keep Part 2 content
7. Transform remaining sections to bilingual format

### Step 4: Update Published Posts List

Add both parts to `_data/published_posts.yml`:

```yaml
cs336:
  - 2025-07-26-cs336-note-train-bpe-tinystories-part1.md
  - 2025-07-26-cs336-note-train-bpe-tinystories-part2.md
```

### Step 5: Verify

1. Check all cross-links work correctly
2. Verify bilingual structure is consistent
3. Ensure code formatting is preserved
4. Test on local Jekyll server: `bundle exec jekyll serve`
5. Review both parts for completeness

## Translation Quality Guidelines

### Content Translation

- **Accuracy**: Translate meaning, not just words
- **Technical Terms**: Use established Chinese technical terminology
- **Code Comments**: Keep code in English, translate only explanatory text
- **Examples**: Translate example text, keep variable names in English

### Formatting Preservation

Must preserve:
- ✅ Code blocks with syntax highlighting
- ✅ Inline code formatting
- ✅ Tables
- ✅ Blockquotes
- ✅ Links (update URLs as needed)
- ✅ Images (with bilingual captions)
- ✅ Mathematical expressions

### Chinese Translation Standards

- Use simplified Chinese (简体中文)
- Keep punctuation appropriate for Chinese text
- Use Chinese quotation marks: 「」『』or ""
- Numbers and units can remain in Arabic numerals
- Technical terms: 词元 (token), 字节 (bytes), 分词器 (tokenizer), etc.

## Common Patterns Reference

### Section Headers

```markdown
## Introduction

## 引言

### Background

### 背景

#### Key Concepts

#### 核心概念
```

### Lists with Technical Content

```markdown
Key features:

主要特性：

- Feature one: Description
- Feature two: Description

- 特性一：说明
- 特性二：说明
```

### Code Examples

```markdown
Here's the implementation:

以下是实现代码：

```python
code here
```

This code demonstrates...

此代码演示了...
```

### Results/Output

```markdown
**Output:**

**输出：**

**Compression ratio:** 3.5x

**压缩比：** 3.5倍
```

## Troubleshooting

### Issue: Cross-links Return 404

**Solution:** Verify link format:
- Must include baseurl: `/chinese-online-content-for-llm/`
- Must follow collection permalink structure
- Must use post title (filename without date), not full date path
- Must end with trailing slash

### Issue: Formatting Breaks After Translation

**Solution:**
- Check that all markdown syntax is preserved
- Verify code block fences are intact
- Ensure heading hierarchy is maintained
- Check for unescaped special characters

### Issue: Inconsistent Section Ordering

**Solution:**
- Always follow English-then-Chinese pattern
- Use the same heading level for both languages
- Keep content pairs together

## Best Practices

1. ✅ **Start with Complete Translation**: Translate the entire post first before splitting
2. ✅ **Preserve Original Structure**: Maintain heading hierarchy and organization
3. ✅ **Consistent Naming**: Use descriptive part titles that indicate content focus
4. ✅ **Test Cross-Links**: Always verify links work in local development
5. ✅ **Update Published List**: Don't forget to add new parts to `published_posts.yml`
6. ✅ **Review Before Publishing**: Check both language versions for completeness

## File Checklist

Before considering a bilingual post complete, verify:

- [ ] Front matter updated with correct title and author
- [ ] Bilingual title structure added
- [ ] Xiaohongshu author box included (if applicable)
- [ ] All sections follow English → Chinese parallel structure
- [ ] Code blocks preserved with correct syntax highlighting
- [ ] Lists converted to bullet points
- [ ] Cross-links use correct relative URL format
- [ ] Post added to `_data/published_posts.yml`
- [ ] Local build succeeds without errors
- [ ] Links tested and working

## Summary

This workflow ensures:
- **Accessibility**: Content available in both English and Chinese
- **Maintainability**: Clear structure for future updates
- **Consistency**: Standardized format across all bilingual posts
- **User Experience**: Easy navigation between related parts
- **Quality**: Preserved formatting and technical accuracy

Follow this guide for all future bilingual post creation and splitting tasks.
