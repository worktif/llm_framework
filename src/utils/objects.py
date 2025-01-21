def deep_join(lst, separator="\n"):
    """
    Recursively flattens a nested list and joins its elements into a single string.

    Parameters:
    - lst (list): The input list, potentially containing nested lists.
    - separator (str): The string used to join the elements. Default is a newline ("\n").

    Returns:
    - str: A single string with all elements of the list joined by the separator.
    """
    # Recursive function to process elements
    def flatten(items):
        for item in items:
            if isinstance(item, list):
                yield from flatten(item)  # Recursion for nested lists
            else:
                yield str(item)  # Convert the item to a string

    # Join all elements using the separator
    return separator.join(flatten(lst))