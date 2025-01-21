import random
import string


def generate_random_word(length=5):
    """
    Generate a random word consisting of lowercase ASCII letters.

    Parameters:
    - length (int): The desired length of the generated word. Default is 5.

    Returns:
    - str: A randomly generated word.
    """
    return ''.join(random.choices(string.ascii_lowercase, k=length))