from textblob import TextBlob

fname = "file_exit.txt"
with open(fname, 'r') as f:
    for line in f:
        
        lang = TextBlob(line).detect_language()
        if (lang != 'en'):
            print(TextBlob(line).translate(to="en"))
        else:
            print(line)
