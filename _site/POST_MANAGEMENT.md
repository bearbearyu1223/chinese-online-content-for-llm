# Post Management Guide

## How to Control Which Posts are Published

This website uses a simple configuration file to control which blog posts appear on the site. This makes it easy to publish or unpublish posts without modifying individual files.

## Configuration File

The published posts are managed through: `_data/published_posts.yml`

## How to Publish a Post

1. Open `_data/published_posts.yml`
2. Add the post filename (including the date prefix) under the `posts:` section
3. Save the file
4. The Jekyll server will automatically regenerate the site

Example:
```yaml
posts:
  - 2025-07-20-cs336-note-get-started.md
  - 2025-07-22-cs336-note-simple-bpe.md
  - 2025-07-26-cs336-note-train-bpe-tinystories.md  # New post added here
```

## How to Unpublish a Post

1. Open `_data/published_posts.yml`
2. Remove the post filename from the list (or comment it out with `#`)
3. Save the file
4. The Jekyll server will automatically regenerate the site

Example:
```yaml
posts:
  - 2025-07-20-cs336-note-get-started.md
  # - 2025-07-22-cs336-note-simple-bpe.md  # Commented out = unpublished
```

## Benefits of This Approach

✅ **Easy to manage**: All post visibility controlled from one file
✅ **No file modifications**: Original post files remain unchanged
✅ **Reversible**: Easily publish/unpublish posts by editing the list
✅ **Version control friendly**: Clear history of what was published when
✅ **Future-proof**: New posts can be added without touching existing ones

## Available Posts

To see all available posts in your repository:
```bash
ls _posts/
```

## Notes

- Posts not in the `published_posts.yml` list will still exist in the `_posts/` folder but won't appear on the website
- You can access unpublished posts directly via their URL if you know it, but they won't be listed on the home page
- The Jekyll RSS feed will also respect this configuration
