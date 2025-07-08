# MyDataAnalyst

MyDataAnalyst is a web app acting like your personal data analyst built with Django and Python. 

- First you upload a .csv file that stores any type of data.
- In the backend app reads this file using pandas python library.
- Then we use the summary of the provided data to ask LLM model via Groq API to generate relevant questions about this dataset. We can select however many questions we want answered or can even add our own.
- Finally we analyze this data (again using Groq) and get answers as well as relevant charts, which get generated with plotly python library.

There is an example of potential datasets in `/datasets/' folder. However, if you like, you can upload your own dataset.
