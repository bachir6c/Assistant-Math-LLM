with open('chat_template_utils.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('.read_text()', '.read_text(encoding="utf-8")')

with open('chat_template_utils.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('patched')