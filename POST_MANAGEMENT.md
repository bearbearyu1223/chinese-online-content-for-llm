# Post Management Guide

## How to Control Which Posts are Published

This website uses a simple configuration file to control which blog posts appear on the site. This makes it easy to publish or unpublish posts without modifying individual files.

## Configuration File

The published posts are managed through: `_data/published_posts.yml`

## Collections Structure

Posts are organized into collections:
- **📚 CS336**: `_cs336/` directory - Stanford CS336 study notes
- **🌱 Career**: `_career/` directory - Career growth and self-reflection
- **🧬 AI Insights**: `_ai-insights/` directory - Latest AI trends and insights

## How to Publish a Post

1. Open `_data/published_posts.yml`
2. Add the post filename (including the date prefix) under the appropriate collection
3. Save the file
4. The Jekyll server will automatically regenerate the site

Example:
```yaml
cs336:
  - 2025-07-20-cs336-note-get-started.md
  - 2025-07-22-cs336-note-simple-bpe.md
  - 2025-07-26-cs336-note-train-bpe-tinystories.md  # New post added here

career:
  - 2025-12-01-my-career-journey.md

ai-insights:
  - 2025-12-15-latest-ai-trends.md
```

## How to Unpublish a Post

1. Open `_data/published_posts.yml`
2. Remove the post filename from the list (or comment it out with `#`)
3. Save the file
4. The Jekyll server will automatically regenerate the site

Example:
```yaml
cs336:
  - 2025-07-20-cs336-note-get-started.md
  # - 2025-07-22-cs336-note-simple-bpe.md  # Commented out = unpublished
```

## Benefits of This Approach

✅ **Easy to manage**: All post visibility controlled from one file
✅ **No file modifications**: Original post files remain unchanged
✅ **Reversible**: Easily publish/unpublish posts by editing the list
✅ **Version control friendly**: Clear history of what was published when
✅ **Future-proof**: New posts can be added without touching existing ones
✅ **Collection-aware**: Manage posts across different collections

## Available Posts

To see all available posts by collection:
```bash
ls _cs336/
ls _career/
ls _ai-insights/
```

## Notes

- Posts not in the `published_posts.yml` list will still exist in their collection folders but won't appear on the website
- You can access unpublished posts directly via their URL if you know it, but they won't be listed on collection pages or the homepage
- The Jekyll RSS feed will also respect this configuration
- Each collection is managed independently in the configuration file
