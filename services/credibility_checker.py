def check_sources(sources_list):
    """
    Evaluates evidence sources based on a known domain whitelist.
    Returns credibility percentage.
    """
    trusted_domains = [
        "bbc", "reuters", "cnn", "ap news", "associated press", "npr", "nytimes", "washington post",
        "wsj", "bloomberg", "the guardian", "al jazeera", "fox news"
    ]
    
    score = 0
    trusted_count = 0
    
    for source_dict in sources_list:
        # Depending on if we passed strings or dicts:
        source_name = source_dict if isinstance(source_dict, str) else source_dict.get("source", "")
        source_lower = source_name.lower()
        
        for trusted in trusted_domains:
            if trusted in source_lower:
                score += 35
                trusted_count += 1
                break
                
    final_score = min(max(score, 15), 98) # Floor at 15 for unknown, ceil at 98
    
    return {
        "score": final_score,
        "verified_count": trusted_count
    }
