from django.shortcuts import render, redirect
from .forms import CSVUploadForm
import pandas as pd
from groq import Groq
from django.conf import settings
import base64
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import re


def get_ai_questions(df_summary):
    client = Groq(api_key=settings.GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""Given the following CSV data summary:

{df_summary}

Provide a list of the most relevant and insightful questions that can be asked about this dataset. 
Separate each question with a unique delimiter, such as '|||'. 
Do not include any introductory or concluding remarks, just the delimited questions.""",
            }
        ],
        model="llama3-8b-8192",  # Using a smaller model for quicker responses
    )
    questions_str = chat_completion.choices[0].message.content
    questions = [
        q.strip() + ("?" if not q.strip().endswith("?") else "")
        for q in questions_str.split("|||")
        if q.strip()
    ]
    return questions


def get_ai_answer(question, df_head, df_columns):
    client = Groq(api_key=settings.GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Given the following question: {question}\n\nAnd the CSV data with head: {df_head}\n\nAnd columns: {df_columns}\n\nProvide a concise answer and suggest a suitable chart type (e.g., bar chart, line chart, scatter plot) if applicable. Format your response as: Answer: <answer>\nChart Type: <chart_type>",
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content


def index(request):
    questions = []
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES["csv_file"]
            try:
                df = pd.read_csv(csv_file)
                request.session["dataframe"] = (
                    df.to_json()
                )  # Store DataFrame in session
                df_summary = (
                    df.head().to_string()
                    + "\n"
                    + str(df.info(verbose=False, buf=None, show_counts=False))
                )  # Simplified summary
                questions = get_ai_questions(df_summary)
                request.session["questions"] = questions  # Store questions in session
                return render(
                    request,
                    "analyzer/index.html",
                    {"message": "File uploaded and processed!", "questions": questions},
                )
            except Exception as e:
                return render(
                    request,
                    "analyzer/index.html",
                    {"error": f"Error processing CSV: {e}"},
                )
    else:
        form = CSVUploadForm()
        # Clear questions from session if not a POST request (i.e., a fresh page load)
        if "questions" in request.session:
            del request.session["questions"]
        questions = []
    return render(
        request, "analyzer/index.html", {"form": form, "questions": questions}
    )


def analyze_data(request):
    if request.method == "POST":
        selected_questions = request.POST.getlist("selected_questions")
        custom_questions_str = request.POST.get("custom_questions", "")
        
        if custom_questions_str:
            custom_questions = [q.strip() for q in custom_questions_str.split('\n') if q.strip()]
            selected_questions.extend(custom_questions)

        if not selected_questions:
            return redirect("index")

        df_json = request.session.get("dataframe")
        if not df_json:
            return redirect("index")

        df = pd.read_json(StringIO(df_json), orient='columns')
        results = []

        for question in selected_questions:
            ai_response = get_ai_answer(
                question, df.head().to_string(), df.columns.tolist()
            )
            answer_match = re.search(r"Answer: (.*)", ai_response)
            chart_type_match = re.search(r"Chart Type: (.*)", ai_response)

            answer = (
                answer_match.group(1).strip()
                if answer_match
                else "No answer generated."
            )
            chart_type = (
                chart_type_match.group(1).strip() if chart_type_match else "None"
            )

            plot_url = None

            def generate_plot(df, chart_type, question, answer):
                plt.style.use('seaborn-v0_8-darkgrid') # Apply a style
                fig, ax = plt.subplots(figsize=(10, 6))
                
                plot_generated = False
                
                try:
                    if chart_type.lower() == "bar chart":
                        # Try to find a categorical and a numerical column
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                        if categorical_cols and numeric_cols:
                            # Use the first categorical and first numeric for a basic bar chart
                            # Limit to top N categories and aggregate the rest into 'Other'
                            N = 20 # Number of top categories to display
                            
                            # Group by categorical column and sum/mean the numeric column
                            # For simplicity, using mean here, but could be sum depending on context
                            grouped_data = df.groupby(categorical_cols[0])[numeric_cols[0]].mean().sort_values(ascending=False)

                            data_to_plot = grouped_data.head(N)
                            
                            # Always use horizontal bar chart for better readability with many categories
                            data_to_plot.plot(kind='barh', ax=ax)
                            ax.set_ylabel(categorical_cols[0])
                            ax.set_xlabel(numeric_cols[0])
                            ax.set_title(f'{numeric_cols[0]} by {categorical_cols[0]} (Top {len(data_to_plot)} Categories)')
                            plot_generated = True
                        elif numeric_cols:
                            # Fallback to distribution of a single numeric column if no categorical
                            df[numeric_cols[0]].value_counts().sort_index().plot(kind='bar', ax=ax)
                            ax.set_title(f'Distribution of {numeric_cols[0]}')
                            ax.set_xlabel(numeric_cols[0])
                            ax.set_ylabel('Count')
                            plot_generated = True

                    elif chart_type.lower() == "line chart":
                        # Try to find a time-series like column and a numerical column
                        # For simplicity, assuming first numeric column against index or another numeric
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        if len(numeric_cols) >= 2:
                            df.plot(x=numeric_cols[0], y=numeric_cols[1], kind='line', ax=ax)
                            ax.set_title(f'{numeric_cols[1]} over {numeric_cols[0]}')
                            ax.set_xlabel(numeric_cols[0])
                            ax.set_ylabel(numeric_cols[1])
                            plot_generated = True
                        elif numeric_cols:
                            df[numeric_cols[0]].plot(kind='line', ax=ax)
                            ax.set_title(f'Trend of {numeric_cols[0]}')
                            ax.set_xlabel('Index')
                            ax.set_ylabel(numeric_cols[0])
                            plot_generated = True

                    elif chart_type.lower() == "scatter plot":
                        # Need at least two numeric columns for a scatter plot
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        if len(numeric_cols) >= 2:
                            df.plot(x=numeric_cols[0], y=numeric_cols[1], kind='scatter', ax=ax)
                            ax.set_title(f'Scatter Plot of {numeric_cols[1]} vs {numeric_cols[0]}')
                            ax.set_xlabel(numeric_cols[0])
                            ax.set_ylabel(numeric_cols[1])
                            plot_generated = True
                    
                    # Add more chart types here (e.g., histogram, pie chart)

                    if plot_generated:
                        plt.tight_layout()
                        buffer = BytesIO()
                        plt.savefig(buffer, format="png")
                        plt.close(fig) # Close the figure to free memory
                        return base64.b64encode(buffer.getvalue()).decode("utf-8")
                except Exception as e:
                    print(f"Error generating plot: {e}")
                    plt.close(fig) # Ensure figure is closed on error
                return None

            plot_url = generate_plot(df, chart_type, question, answer)

            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "chart_type": chart_type,
                    "plot_url": plot_url,
                }
            )
        return render(request, "analyzer/analysis.html", {"results": results})
    return redirect("index")
