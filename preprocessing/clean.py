def remove_symbols(text):
    symbols = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    translator = str.maketrans('', '', symbols)
    return text.translate(translator)