import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the feedback data
def load_data():
    try:
        # Load CSV and ensure proper column names
        df = pd.read_csv("feedback.csv", names=["Timestamp", "Sentence", "Prediction", "Feedback"], header=0)
        return df
    except FileNotFoundError:
        st.warning("No feedback data found!")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=["Timestamp", "Sentence", "Prediction", "Feedback"])


# Main dashboard
def main():
    st.title("Cultural Appropriateness Bot - Dashboard")
    st.sidebar.header("Filters")
    data = load_data()

    if not data.empty:
        # Sidebar Filters
        date_filter = st.sidebar.date_input("Filter by Date")
        prediction_filter = st.sidebar.multiselect(
            "Filter by Prediction", ["Appropriate", "Inappropriate", "Neutral"]
        )

        # Apply Filters
        if date_filter:
            data["Timestamp"] = pd.to_datetime(data["Timestamp"])
            data = data[data["Timestamp"].dt.date == date_filter]

        if prediction_filter:
            data = data[data["Prediction"].isin(prediction_filter)]

        # Feedback Overview: Agree vs. Disagree
        st.header("Feedback Overview")
        feedback_counts = data["Feedback"].value_counts()

        if not feedback_counts.empty:
            fig, ax = plt.subplots()

            # Bar chart with feedback counts
            bars = ax.bar(feedback_counts.index, feedback_counts.values, color=["#4CAF50", "#FF5252"])

            # Add data labels (counts and percentages)
            for bar in bars:
                count = bar.get_height()
                percentage = (count / feedback_counts.sum()) * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{count} ({percentage:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black"
                )

            # Enhance chart aesthetics
            ax.set_title("Feedback: Agree Vs Disagree", fontsize=16, pad=30)
            ax.set_ylabel("Number of Feedback Entries", fontsize=12)
            ax.set_xlabel("Feedback Type", fontsize=12)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.set_axisbelow(True)

            # Display the chart
            st.pyplot(fig)
        else:
            st.write("No feedback data available for selected filters.")


        # Pie Chart: Prediction Distribution
        st.header("Prediction Breakdown")
        prediction_counts = data["Prediction"].value_counts()
        if not prediction_counts.empty:
            fig2, ax2 = plt.subplots()
            ax2.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title("Prediction Distribution")
            st.pyplot(fig2)
        else:
            st.write("No prediction data available for selected filters.")

        # Feedback Table
        st.header("Recent Feedback")
        st.dataframe(data[["Timestamp", "Sentence", "Prediction", "Feedback"]])


        # Prediction Breakdown by Feedback
        st.header("Prediction Breakdown by Feedback")
        breakdown_data = data.groupby(["Prediction", "Feedback"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots()
        breakdown_data.plot(kind="bar", stacked=True, ax=ax, color=["#4CAF50", "#FF5252"])
        ax.set_title("Prediction Breakdown by Feedback")
        ax.set_ylabel("Number of Feedback Entries")
        ax.set_xlabel("Prediction Type")
        st.pyplot(fig)


    else:
        st.info("No feedback data available yet.")

    st.download_button(
        label="Downlaod Feedback Data",
        data=data.to_csv(index=False),
        file_name="feedback_data.csv",
        mime="text/csv",
    )
        


if __name__ == "__main__":
    main()
