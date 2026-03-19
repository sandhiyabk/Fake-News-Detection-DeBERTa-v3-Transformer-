def detect_bias(text):
    """
    Detects sensationalism and bias using keyword heuristics.
    Returns a bias score and level.
    """
    bias_words = [
        "shocking", "exposed", "truth", "must see", "secret", "they don't want you to know",
        "mind-blowing", "scandal", "destroy", "libtard", "snowflake", "fake", "hoax",
        "conspiracy", "miracle", "banned", "cover-up", "insane", "unbelievable"
    ]
    
    text_lower = text.lower()
    matches = [word for word in bias_words if word in text_lower]
    score = len(matches)
    
    if score == 0:
        level = "Low"
    elif score <= 2:
        level = "Medium"
    else:
        level = "High"
        
    # Scale score to a rough percentage for the UI
    percentage = min(score * 15, 100)
    
    return {
        "score": percentage,
        "level": level,
        "matched_keywords": matches
    }
