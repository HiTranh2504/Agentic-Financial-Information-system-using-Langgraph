import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

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
