from spellchecker import SpellChecker
 
spell = SpellChecker()

def check(text):
    misspelled = spell.unknown(text)
    for word in misspelled:
        corrected_word = spell.correction(word)
        if corrected_word:
            text[text.index(word)] = corrected_word
    return text