#!/usr/bin/env python3
"""
Script to prepare and convert markdown files to PDF.
Converts Jekyll-style absolute paths and generates PDF using markdown-pdf.

Usage:
    python3 prepare-for-pdf.py _posts/2025-07-20-cs336-note-get-started.md
"""

import sys
import re
import subprocess
from pathlib import Path

def convert_image_paths(content):
    """
    Convert Jekyll-style absolute paths (/assets/...) to workspace-relative paths (assets/...)
    Also converts external xiaohongshu logo URL to local path for PDF rendering.
    This works when the markdown file is in the workspace root.
    """
    # Pattern 1: ![alt text](/assets/path/to/image.png)
    pattern1 = r'!\[([^\]]*)\]\(/assets/([^\)]+)\)'

    def replace_jekyll_path(match):
        alt_text = match.group(1)
        asset_path = match.group(2)
        # Remove leading slash for workspace-relative path
        relative_path = f'assets/{asset_path}'
        return f'![{alt_text}]({relative_path})'

    converted_content = re.sub(pattern1, replace_jekyll_path, content)

    # Pattern 2: Replace external xiaohongshu logo URL with local path
    xiaohongshu_pattern = r'src="https://static\.cdnlogo\.com/logos/r/77/rednote-xiaohongshu\.svg"'
    converted_content = re.sub(
        xiaohongshu_pattern,
        'src="assets/picture/xiaohongshu-logo.svg"',
        converted_content
    )

    # Pattern 3: Replace HTML author box with simple markdown for PDF
    # Match the entire <style>...</style> block and the <div>...</div> author box
    author_box_pattern = r'<style>.*?</style>\s*<div style="padding:12px.*?</div>'
    markdown_author = '''**大模型我都爱**

**小红书：** ![小红书](assets/picture/xiaohongshu-logo.svg) [美女我都爱](https://www.xiaohongshu.com/user/profile/5b2c5758e8ac2b08bf20e38d)

**IP属地：** 美国

---'''

    converted_content = re.sub(author_box_pattern, markdown_author, converted_content, flags=re.DOTALL)

    # Pattern 4: Remove Jekyll front matter (YAML between --- markers)
    front_matter_pattern = r'^---\s*\n.*?\n---\s*\n'
    converted_content = re.sub(front_matter_pattern, '', converted_content, flags=re.DOTALL)

    return converted_content

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 prepare-for-pdf.py <markdown_file>")
        print("Example: python3 prepare-for-pdf.py _posts/2025-07-20-cs336-note-get-started.md")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)

    # Read the original content
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Convert image paths
    converted_content = convert_image_paths(content)

    # Create output filenames in ROOT directory
    temp_md = Path(f"{input_file.stem}_pdf{input_file.suffix}")
    output_pdf = Path(f"{input_file.stem}.pdf")

    # Write the converted content to root directory
    with open(temp_md, 'w', encoding='utf-8') as f:
        f.write(converted_content)

    # Count conversions
    jekyll_paths = content.count('/assets/')
    xiaohongshu_logos = content.count('https://static.cdnlogo.com/logos/r/77/rednote-xiaohongshu.svg')
    author_boxes = content.count('<style>')  # Count HTML author boxes

    print(f"✓ Created PDF-friendly markdown: {temp_md}")
    if jekyll_paths > 0:
        print(f"  Converted {jekyll_paths} Jekyll image path(s)")
    if xiaohongshu_logos > 0:
        print(f"  Converted {xiaohongshu_logos} external xiaohongshu logo(s) to local path")
    if author_boxes > 0:
        print(f"  Replaced {author_boxes} HTML author box(es) with markdown")
    print(f"\n🔄 Converting to PDF...")

    # Check if markdown-pdf is installed
    try:
        subprocess.run(['which', 'markdown-pdf'],
                      check=True,
                      capture_output=True)
    except subprocess.CalledProcessError:
        print("\n❌ markdown-pdf not found!")
        print("Install it with: npm install -g markdown-pdf")
        print(f"\nThen run: markdown-pdf {temp_md} -o {output_pdf}")
        sys.exit(1)

    # Convert to PDF using markdown-pdf
    # Check if custom CSS exists
    css_file = Path('pdf-no-breaks.css')
    cmd = ['markdown-pdf', str(temp_md), '-o', str(output_pdf)]
    if css_file.exists():
        cmd.extend(['-s', str(css_file)])
        print(f"  Using custom CSS: {css_file}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=120
        )

        print(f"✅ PDF created successfully: {output_pdf}")

        # Get file size
        pdf_size = output_pdf.stat().st_size
        if pdf_size < 1024:
            size_str = f"{pdf_size} bytes"
        elif pdf_size < 1024 * 1024:
            size_str = f"{pdf_size / 1024:.1f} KB"
        else:
            size_str = f"{pdf_size / (1024 * 1024):.1f} MB"

        print(f"  Size: {size_str}")
        print(f"\n🧹 Cleaning up temporary file: {temp_md}")
        temp_md.unlink()

        print(f"\n✨ Done! Your PDF is ready: {output_pdf}")

    except subprocess.TimeoutExpired:
        print("\n❌ PDF conversion timed out")
        print(f"You can try manually: markdown-pdf {temp_md} -o {output_pdf}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ PDF conversion failed")
        print(f"Error: {e.stderr if e.stderr else 'Unknown error'}")
        print(f"\nYou can try manually: markdown-pdf {temp_md} -o {output_pdf}")
        sys.exit(1)

if __name__ == "__main__":
    main()
