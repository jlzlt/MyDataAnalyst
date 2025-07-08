from django.shortcuts import render, redirect
from .forms import CSVUploadForm
import pandas as pd
from groq import Groq
from django.conf import settings
from io import StringIO
import plotly.express as px
import plotly.io as pio
import re
import json


def get_ai_questions(df_summary):
    client = Groq(api_key=settings.GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f'"""Given the following CSV data summary:\n\n{df_summary}\n\nProvide a list of insightful questions about this dataset. Output ONLY the questions, each separated by \'|||\'. Do not include any other text or formatting."""',
            }
        ],
        model="llama3-8b-8192",  # Using a smaller model for quicker responses
    )
    questions_str = chat_completion.choices[0].message.content
    # Use regex to split by the delimiter OR newlines for more robust parsing
    questions = [
        q.strip() + ("?" if not q.strip().endswith("?") else "")
        for q in re.split(r'\|\|\||\n', questions_str)
        if q.strip()
    ]
    return questions


def get_ai_answer(question, df_head, df_columns):
    client = Groq(api_key=settings.GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f'"""For the question: {question}\nGiven the CSV data head: {df_head}\nAnd columns: {df_columns}\n\nProvide a concise answer, suggest a suitable chart type (e.g., bar chart, line chart, scatter plot), and suggest specific columns for plotting (e.g., x_column, y_column, hue_column). Respond ONLY with a JSON object with keys: \'answer\', \'chart_type\', and \'plot_columns\' (a dictionary). Example: {{"answer": "The average beer servings is X.", "chart_type": "bar chart", "plot_columns": {{"x": "country", "y": "beer_servings"}}}}."""',
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
            custom_questions = [
                q.strip() for q in custom_questions_str.split("\n") if q.strip()
            ]
            selected_questions.extend(custom_questions)

        if not selected_questions:
            return redirect("index")

        df_json = request.session.get("dataframe")
        if not df_json:
            return redirect("index")

        df = pd.read_json(StringIO(df_json), orient="columns")
        results = []

        for question in selected_questions:
            ai_response_str = get_ai_answer(
                question, df.head().to_string(), df.columns.tolist()
            )
            print(f"Raw AI response for answer: {ai_response_str}")  # Debug print

            answer = "No answer generated."
            chart_type = "None"
            plot_columns = {}

            try:
                ai_response = json.loads(ai_response_str)
                answer = ai_response.get("answer", answer)
                chart_type = ai_response.get("chart_type", chart_type)
                plot_columns = ai_response.get("plot_columns", plot_columns)
                print(
                    f"Parsed AI response: Answer='{answer}', Chart Type='{chart_type}', Plot Columns='{plot_columns}'"
                )  # Debug print
            except json.JSONDecodeError:
                print(
                    f"JSONDecodeError: Could not parse AI response as JSON. Raw response: {ai_response_str}"
                )  # Debug print
                # Fallback if AI doesn't return valid JSON
                answer_match = re.search(r"Answer: (.*)", ai_response_str)
                chart_type_match = re.search(r"Chart Type: (.*)", ai_response_str)
                answer = answer_match.group(1).strip() if answer_match else answer
                chart_type = (
                    chart_type_match.group(1).strip()
                    if chart_type_match
                    else chart_type
                )

            plot_div = None

            def generate_plot(df, chart_type, question, answer, plot_columns):
                fig = None
                try:
                    x_col = plot_columns.get("x")
                    y_col = plot_columns.get("y")
                    hue_col = plot_columns.get("hue")

                    # Validate columns exist in DataFrame
                    if x_col and x_col not in df.columns:
                        x_col = None
                    if y_col and y_col not in df.columns:
                        y_col = None
                    if hue_col and hue_col not in df.columns:
                        hue_col = None

                    title = f"{y_col} by {x_col}" if x_col and y_col else question

                    if chart_type.lower() == "bar chart":
                        if x_col and y_col:
                            fig = px.bar(
                                df, x=x_col, y=y_col, color=hue_col, title=title
                            )
                        elif y_col:  # Fallback for single numeric column
                            fig = px.histogram(
                                df, x=y_col, title=f"Distribution of {y_col}"
                            )

                    elif chart_type.lower() == "line chart":
                        if x_col and y_col:
                            fig = px.line(
                                df,
                                x=x_col,
                                y=y_col,
                                color=hue_col,
                                title=title,
                                markers=True,
                            )
                        elif y_col:  # Fallback for single numeric column
                            fig = px.line(
                                df, y=y_col, title=f"Trend of {y_col}", markers=True
                            )

                    elif chart_type.lower() == "scatter plot":
                        if x_col and y_col:
                            fig = px.scatter(
                                df, x=x_col, y=y_col, color=hue_col, title=title
                            )

                    if fig:
                        fig.update_layout(template="plotly_dark")
                        return pio.to_html(fig, full_html=False)

                except Exception as e:
                    print(f"Error generating plot: {e}")
                return None

            plot_div = generate_plot(df, chart_type, question, answer, plot_columns)

            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "chart_type": chart_type,
                    "plot_div": plot_div,
                }
            )
        return render(request, "analyzer/analysis.html", {"results": results})
    return redirect("index")
