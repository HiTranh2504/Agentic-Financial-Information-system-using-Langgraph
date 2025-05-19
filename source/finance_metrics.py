import yaml
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

# METADATA_PATH = os.getenv("METADATA_PATH", "source\\financial_metadata.yml")


METADATA_PATH = str(Path(r"D:/HK2_2024_2025/Data Platform/Thuc_hanh/CK/chatbot-financial-langgraph/source/financial_metadata.yml"))


def load_formulas():
    with open(METADATA_PATH, "r") as f:
        return yaml.safe_load(f)


# def identify_metric(question, metadata):
#     question = question.lower()
#     for category in metadata:
#         for metric in metadata[category]:
#             if metric.lower() in question:
#                 return category, metric
#     return None, None

def identify_metric(question, metadata):
    # question = question.lower()
    question = question.lower().strip()
    question = question.replace("?", "").replace(",", "").replace(".", "")


    # 1. Ưu tiên match chính xác tên metric
    for category in metadata:
        for metric in metadata[category]:
            if metric.lower() in question:
                return category, metric

    # 2. Mapping alias từ các cụm từ sang metric_name
    keyword_aliases = {
        "market capitalization": "MarketCap",
        "market cap": "MarketCap",
        "p/e": "PE",
        "price to earnings": "PE",
        "roe": "ROE",
        "dividend": "TotalDividends",
        "dividends": "TotalDividends",
        "returns": "DailyReturn",
        "cumulative return": "CumulativeReturn",
        "rsi": "RSI",
        "macd": "MACD",
        "moving average": "MA",
        "rolling average": "MA",
        "bollinger": "BollingerBands",
        "volume": "Volume",
        "high low range": "HighLowRange",
        "scatter plot": "MarketCap_vs_PE",  # gán mặc định nếu chưa rõ
        "correlation": "CorrelationMatrix",
        "sector distribution": "SectorDistribution",
        "closing price": "Close",
        "close": "Close",
        "stock price": "Close",
        "price chart": "Close",
        "plot": "Close",
    }

    for phrase, metric_name in keyword_aliases.items():
        if phrase in question:
            for category in metadata:
                if metric_name in metadata[category]:
                    return category, metric_name

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
        elif metric_name == "MarketCap":
            return data["market_cap"]
        elif metric_name == "Volume":
            return data["volume"]
        elif metric_name == "Close":
            return data["Close"] if "Close" in data else data["close"]
        elif metric_name == "TotalDividends":
            return data["dividends"]
        elif metric_name in ["RSI", "MACD", "MA", "BollingerBands"]:
            return f"[Simulated {metric_name} calculation from price_series]"
        return "Unsupported metric"
    except Exception as e:
        return f"Error computing {metric_name}: {str(e)}"

