import yaml
import os
from dotenv import load_dotenv

load_dotenv()

METADATA_PATH = os.getenv("METADATA_PATH", "financial_metadata.yml")


def load_formulas():
    with open(METADATA_PATH, "r") as f:
        return yaml.safe_load(f)


def identify_metric(question, metadata):
    question = question.lower()
    for category in metadata:
        for metric in metadata[category]:
            if metric.lower() in question:
                return category, metric
    return None, None


def get_required_fields(metric_category, metric_name, metadata):
    return metadata[metric_category][metric_name].get("required_fields", [])


def compute_metric(metric_name, data):
    try:
        if metric_name == "EPS":
            return data["net_income"] / data["shares_outstanding"]

        elif metric_name == "PE":
            return data["stock_price"] / data["eps"]

        elif metric_name == "ROE":
            return data["net_income"] / data["equity"]

        elif metric_name == "DebtRatio":
            return data["total_liabilities"] / data["total_assets"]

        # Placeholder cho các chỉ số TA
        elif metric_name in ["RSI", "MACD", "MA", "BollingerBands"]:
            return f"[Simulated {metric_name} calculation from price_series]"

        return "Unsupported metric"
    except Exception as e:
        return f"Error computing {metric_name}: {str(e)}"
