import sys
import os
import re
import random
import nltk
from collections import Counter
from langdetect import detect
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("wordnet")


def text_from_file(filepath):
    if not os.path.isfile(filepath):
        print(f"Error: {filepath} not found")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


def calculate_word_frequencies(text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    return word_counts


def stylometry(text):
    text_no_spaces = text.strip()
    char_count = len(text_no_spaces)

    words = re.findall(r'\b\w+\b', text_no_spaces)
    word_count = len(words)

    word_frequencies = calculate_word_frequencies(text_no_spaces)

    try:
        detected_language = detect(text_no_spaces)
    except Exception:
        detected_language = "Unknown (could not detect)"

    print("\nStylometric Analysis:")
    print("-" * 50)
    print(f"Language Detected: {detected_language}")
    print(f"Length in characters: {char_count}")
    print(f"Length in words: {word_count}")
    print("\nMost Frequent Words:")
    for word, freq in word_frequencies.most_common(10):
        print(f"  {word}: {freq}")


def get_replacements(word):
    replacements = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            replacements.add(lemma.name().replace("_", " ").lower())
            if lemma.antonyms():
                replacements.add(lemma.antonyms()[0].name().replace("_", " ").lower())

        hypernyms = syn.hypernyms()
        hyponyms = syn.hyponyms()
        replacements.update(h.name().split(".")[0].replace("_", " ").lower() for h in hypernyms)
        replacements.update(h.name().split(".")[0].replace("_", " ").lower() for h in hyponyms)

    replacements.discard(word.lower())
    return list(replacements)


def replace_words(text, replacement_ratio=0.2):
    words = word_tokenize(text)
    replacement_ratio_rand = random.uniform(replacement_ratio, 0.8)
    num_replacements = max(1, int(len(words) * replacement_ratio_rand))
    indices_to_replace = random.sample(range(len(words)), num_replacements)

    for i in indices_to_replace:
        word = words[i]
        replacements = get_replacements(word.lower())
        if replacements:
            words[i] = random.choice(replacements)

    return " ".join(words)


def generate_alternatives(text, num_versions=5, replacement_ratio=0.2):
    print("\nAlternative Text Version:")
    print("-" * 50)
    for i in range(num_versions):
        print(f"\nVersion {i + 1}:")
        print("-" * 50)
        print(replace_words(text, replacement_ratio))


def get_related_words(keyword):
    related_words = set()

    if keyword.lower() in ["birds", "rodents"]:
        related_words.add(keyword.lower())

    for syn in wordnet.synsets(keyword):
        for lemma in syn.lemmas():
            related_words.add(lemma.name().replace("_", " ").lower())
            if lemma.antonyms():
                for antonym in lemma.antonyms():
                    related_words.add(antonym.name().replace("_", " ").lower())

        hypernyms = syn.hypernyms()
        hyponyms = syn.hyponyms()
        related_words.update(h.name().split(".")[0].replace("_", " ").lower() for h in hypernyms)
        related_words.update(h.name().split(".")[0].replace("_", " ").lower() for h in hyponyms)

    return list(related_words)


def find_keywords(text, keywords):
    sentences = sent_tokenize(text)
    keyword_contexts = {}

    for keyword in keywords:
        related_words = get_related_words(keyword)

        pattern = r'\b(?:' + '|'.join(map(re.escape, related_words)) + r')\b'

        for sentence in sentences:
            match = re.search(pattern, sentence)
            if match:
                negation_pattern = r'\b(?:not|no|never|without|hardly|rarely)\b.*?\b' + re.escape(
                    match.group(0)) + r'\b'
                is_negated = bool(re.search(negation_pattern, sentence))

                if keyword not in keyword_contexts:
                    keyword_contexts[keyword] = []
                if is_negated:
                    keyword_contexts[keyword].append(f"(Negated) {sentence}")
                else:
                    keyword_contexts[keyword].append(sentence)

    return keyword_contexts


def generate_sentence_with_keyword(keyword):
    keyword = keyword.upper()
    templates = [
        f"The {keyword} behavior is often observed in animals.",
        f"Being {keyword} can sometimes be an advantage.",
        f"In certain situations, {keyword} traits are essential.",
        f"The environment plays a key role in shaping {keyword} tendencies.",
        f"Scientists studied how {keyword} characteristics impact survival.",
        f"In nature, {keyword} animals tend to adapt differently.",
    ]
    sentence = random.choice(templates)

    return sentence
