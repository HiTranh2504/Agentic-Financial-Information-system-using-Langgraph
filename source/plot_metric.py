import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64
import io

def detect_moving_average_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        col_lc = col.lower()
        if "rolling" in col_lc or "average" in col_lc or "ma" in col_lc:
            return col
    return None


def plot_chart(df, chart_type, question):

    
    try:
        plt.figure(figsize=(10, 5))

        if chart_type == "line":
            plt.plot(df["Date"], df["Close"], label="Close Price", color="blue")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.title(question)
            plt.xticks(rotation=45)
            plt.legend()

        elif chart_type == "line_ma":
            plt.plot(df["Date"], df["Close"], label="Close Price", color="blue")

            ma_col = detect_moving_average_column(df)
            if ma_col:
                plt.plot(df["Date"], df[ma_col], label=ma_col.replace("_", " "), linestyle="--", color="orange")
            else:
                print("⚠️ Không tìm thấy cột rolling average.")
            
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.title(question)
            plt.xticks(rotation=45)
            plt.legend()
        
        elif chart_type == "box":
            df = df.sort_values("Date")
            
            # ✅ Tìm cột daily return có sẵn (nếu LLM đã tính)
            return_col = None
            for col in df.columns:
                if "return" in col.lower():
                    return_col = col
                    break

            # ✅ Nếu không có → tự tính từ Close
            if return_col is None:
                if "Close" in df.columns:
                    df["return"] = df["Close"].pct_change()
                    return_col = "return"
                else:
                    raise ValueError("⚠️ Không tìm thấy cột return hoặc Close để vẽ boxplot.")

            plt.boxplot(df[return_col].dropna())
            plt.title(question)




        elif chart_type == "cumulative":
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            df = df.sort_values("Date")
            base_price = df["Close"].iloc[0]
            df["Cumulative Return"] = df["Close"] / base_price - 1
            plt.plot(df["Date"], df["Cumulative Return"], label="Cumulative Return", color="green")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.title(question)
            plt.xticks(rotation=45)
            plt.legend()

        elif chart_type == "range":
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            df = df.sort_values("Date")
            df["Range"] = df["High"] - df["Low"]
            plt.plot(df["Date"], df["Range"], label="High-Low Range", color="purple")
            plt.xlabel("Date")
            plt.ylabel("Price Range")
            plt.title(question)
            plt.xticks(rotation=45)
            plt.legend()

        elif chart_type == "dividend":
            if "Dividends" not in df.columns:
                raise ValueError("Data missing Dividends column")
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            df = df.sort_values("Date")
            df["Cumulative Dividends"] = df["Dividends"].cumsum()
            plt.plot(df["Date"], df["Cumulative Dividends"], label="Cumulative Dividends", color="orange")
            plt.xlabel("Date")
            plt.ylabel("Dividends")
            plt.title(question)
            plt.xticks(rotation=45)
            plt.legend()



        elif chart_type == "bar":
            x_col = df.columns[0]
            y_col = df.columns[1]
            plt.bar(df[x_col], df[y_col], color="skyblue")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(question)
            plt.xticks(rotation=45)

        elif chart_type == "pie":
            labels = df.iloc[:, 0]
            values = df.iloc[:, 1]
            plt.pie(values, labels=labels, autopct='%1.1f%%')
            plt.title(question)

        elif chart_type == "scatter":
            x_col, y_col = df.columns[:2]
            plt.scatter(df[x_col], df[y_col], alpha=0.7)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(question)

        elif chart_type == "hist":
            val_col = df.select_dtypes(include='number').columns[0]
            plt.hist(df[val_col], bins=20, color="orange", edgecolor="black")
            plt.xlabel(val_col)
            plt.title(question)

        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include='number')
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")

        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return image_base64

    except Exception as e:
        print(f"❌ Lỗi vẽ biểu đồ ({chart_type}): {e}")
        return None
