import pandas as pd

def clean_value(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() in ["nan", "none", "null", "[deleted]", "[removed]"]:
        return ""
    return value
