# PDF Export - Quick Reference

## Generate PDF (One Command)

```bash
python3 prepare-for-pdf.py _posts/YOUR_POST.md
```

## Examples

```bash
# First tutorial
python3 prepare-for-pdf.py _posts/2025-07-20-cs336-note-get-started.md

# BPE tutorial
python3 prepare-for-pdf.py _posts/2025-07-22-cs336-note-simple-bpe.md

# All CS336 posts
for file in _posts/*cs336*.md; do python3 prepare-for-pdf.py "$file"; done
```

## Page Break Control

✅ **Automatic** - The script uses `pdf-no-breaks.css` to minimize page breaks

### What it prevents:
- ✅ Breaks inside code blocks
- ✅ Breaks inside tables
- ✅ Breaks inside lists
- ✅ Breaks between headings and content
- ✅ Orphans and widows

### Customize CSS:
```bash
# Edit to adjust page break behavior
code pdf-no-breaks.css
```

### Disable CSS:
```bash
# Temporarily disable
mv pdf-no-breaks.css pdf-no-breaks.css.backup

# Generate PDF
python3 prepare-for-pdf.py _posts/YOUR_POST.md

# Re-enable
mv pdf-no-breaks.css.backup pdf-no-breaks.css
```

## Files Generated

- **Output:** `POST_NAME.pdf` (in repository root)
- **Temp:** `POST_NAME_pdf.md` (auto-cleaned)
- **Ignored:** Both files are in `.gitignore`

## First Time Setup

```bash
# Install markdown-pdf (only needed once)
npm install -g markdown-pdf
```

## Troubleshooting

### No page breaks at all?
- Edit `pdf-no-breaks.css`
- Comment out some `page-break-inside: avoid` rules
- Or delete the CSS file to use default styling

### Need more control?
- See full documentation: `PDF_EXPORT_GUIDE.md`
