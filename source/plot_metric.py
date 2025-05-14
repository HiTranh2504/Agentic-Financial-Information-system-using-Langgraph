import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd

def plot_metric(metric_name: str, df: pd.DataFrame) -> str:
    """
    Vẽ biểu đồ tùy theo metric TA và trả về hình ảnh dưới dạng base64
    """
    try:
        prices = df["price"].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10, 5))

        if metric_name == "MA":
            ma = prices.rolling(window=20).mean()
            ax.plot(prices, label="Price")
            ax.plot(ma, label="MA 20")
            ax.set_title("Moving Average")

        elif metric_name == "BollingerBands":
            ma = prices.rolling(window=20).mean()
            std = prices.rolling(window=20).std()
            upper = ma + (2 * std)
            lower = ma - (2 * std)
            ax.plot(prices, label="Price")
            ax.plot(ma, label="MA 20")
            ax.fill_between(range(len(prices)), lower, upper, color="gray", alpha=0.3, label="Bollinger Bands")
            ax.set_title("Bollinger Bands")

        elif metric_name == "RSI":
            delta = prices.diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = -delta.clip(upper=0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            ax.plot(rsi, label="RSI")
            ax.axhline(70, color='red', linestyle='--')
            ax.axhline(30, color='green', linestyle='--')
            ax.set_title("RSI Indicator")

        elif metric_name == "MACD":
            ema12 = prices.ewm(span=12, adjust=False).mean()
            ema26 = prices.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            ax.plot(macd, label="MACD")
            ax.set_title("MACD Indicator")

        else:
            return ""

        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return image_base64

    except Exception as e:
        print(f"Lỗi vẽ biểu đồ {metric_name}: {e}")
        return ""
