import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the feedback data
def load_data():
    try:
        # Load CSV and ensure proper column names
        df = pd.read_csv("feedback.csv", names=["Timestamp", "Sentence", "Prediction", "Feedback"], header=0)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Convert timestamp to datetime
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
            "Filter by Prediction", ["Appropriate", "Inappropriate", "Neutral", "Invalid Input"]
        )

        # Apply Filters
        if date_filter:
            data = data[data["Timestamp"].dt.date == date_filter]

        if prediction_filter:
            data = data[data["Prediction"].isin(prediction_filter)]

        if data.empty:
            st.warning("No data available for the selected filters!")
        else:
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
                ax.set_title("Feedback: Agree vs Disagree", fontsize=16, pad=20)
                ax.set_ylabel("Number of Feedback Entries", fontsize=12)
                ax.set_xlabel("Feedback Type", fontsize=12)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                ax.set_axisbelow(True)

                # Display the chart
                st.pyplot(fig)
            else:
                st.info("No feedback data available for selected filters.")

            # Bar Chart: Prediction Distribution
            st.header("Prediction Breakdown")
            prediction_counts = data["Prediction"].value_counts()
            if not prediction_counts.empty:
                fig3, ax3 = plt.subplots()
                bars = ax3.bar(
                    prediction_counts.index, 
                    prediction_counts.values, 
                    color=["#FF5252", "#4CAF50", "#FFCC00", "#CCCCCC"]
                )
                
                # Add labels to bars
                for bar in bars:
                    count = bar.get_height()
                    percentage = (count / prediction_counts.sum()) * 100
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{count} ({percentage:.1f}%)",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color="black"
                    )
                
                # Adjust x-axis label alignment
                ax3.set_xticks(range(len(prediction_counts.index)))
                ax3.set_xticklabels(prediction_counts.index, rotation=45, ha="right", fontsize=10)

                ax3.set_title("Prediction Distribution", fontsize=16, pad=20)
                ax3.set_ylabel("Number of Predictions", fontsize=12)
                ax3.set_xlabel("Prediction Type", fontsize=12)
                ax3.grid(axis="y", linestyle="--", alpha=0.7)
                ax3.set_axisbelow(True)

                st.pyplot(fig3)
            else:
                st.info("No prediction data available for selected filters.")


            # Feedback Table
            st.header("Recent Feedback")
            st.dataframe(data[["Timestamp", "Sentence", "Prediction", "Feedback"]])

            # Prediction Breakdown by Feedback
            st.header("Prediction Breakdown by Feedback")
            breakdown_data = data.groupby(["Prediction", "Feedback"]).size().unstack(fill_value=0)
            fig, ax = plt.subplots()
            breakdown_data.plot(kind="bar", stacked=True, ax=ax, color=["#4CAF50", "#FF5252"])
            ax.set_title("Prediction Breakdown by Feedback", fontsize=14)
            ax.set_ylabel("Number of Feedback Entries", fontsize=12)
            ax.set_xlabel("Prediction Type", fontsize=12)
            st.pyplot(fig)

    else:
        st.info("No feedback data available yet.")

    # Download button
    st.download_button(
        label="Download Feedback Data",
        data=data.to_csv(index=False),
        file_name="feedback_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
