import re

def is_valid_comment(comment):
    """
    is_valid_comment(str comment)

    The function takes in a string and returns True or False for the string passes to be a valid string for this project.
    
    What a valid comment looks like:
    1. Character length more than 20
    2. Does not contain bot words like "www", "http", "https". Characters often used by bots to make comments.
    3. Does not contain alphanumeric charcters like E500, C0111.
    4. Does not contain mentions of tools like "[flake8]", "[pylint]".

    input: string
    output: boolean
    """
    if len(comment) < 20:
        return False
    
    bot_words = ["http", "https", "www"]
    if any(i in comment for i in bot_words):
        return False
    
    if re.search(r"[A-Z]\d+", comment):
        return False
    
    if re.search(r"\[\w+\]", comment):
        return False
    
    return True
