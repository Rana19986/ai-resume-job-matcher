from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import io


app = FastAPI(
    title="AI Resume Job Matcher"
)


SKILLS = [
    "Python", "Java", "C++", "JavaScript",
    "SQL", "MySQL", "PostgreSQL", "MongoDB",
    "NumPy", "Pandas", "Matplotlib", "Seaborn",
    "Scikit-learn", "TensorFlow", "PyTorch",
    "Keras", "XGBoost", "LightGBM", "CatBoost",
    "Machine Learning", "Deep Learning",
    "Artificial Intelligence",
    "Natural Language Processing",
    "Computer Vision", "NLP",
    "Feature Engineering", "Data Preprocessing",
    "Data Analysis", "Statistics", "Data Science",
    "FastAPI", "Flask", "Django",
    "REST API", "Docker", "Kubernetes",
    "Git", "GitHub", "Linux",
    "AWS", "Google Cloud", "Azure", "GCP",
    "Hugging Face", "Transformers", "BERT",
    "LLM", "Generative AI", "OpenAI",
    "LangChain", "RAG",
    "Apache Spark", "React", "Node.js"
]


def clean_text(text):
    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text):

    text_lower = text.lower()
    found = []

    for skill in SKILLS:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill.lower())
            + r"(?!\w)"
        )

        if re.search(pattern, text_lower):
            found.append(skill)

    return sorted(set(found))


def extract_pdf_text(data):

    reader = PdfReader(
        io.BytesIO(data)
    )

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def tfidf_score(resume, job):

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )

        matrix = vectorizer.fit_transform(
            [resume, job]
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return float(score * 100)

    except Exception:

        return 0.0


def keyword_score(resume, job):

    resume_words = set(
        re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9+#.-]*\b",
            resume.lower()
        )
    )

    job_words = set(
        re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9+#.-]*\b",
            job.lower()
        )
    )

    stopwords = {
        "the", "and", "or", "for",
        "with", "this", "that",
        "from", "your", "you",
        "our", "are", "will",
        "have", "has", "job",
        "role", "work", "years",
        "year", "to", "of",
        "in", "on", "a", "an"
    }

    resume_words -= stopwords
    job_words -= stopwords

    if not job_words:
        return 0.0

    matching_words = (
        resume_words & job_words
    )

    return (
        len(matching_words)
        / len(job_words)
        * 100
    )


def analyze(resume, job):

    resume = clean_text(resume)
    job = clean_text(job)

    resume_skills = set(
        extract_skills(resume)
    )

    job_skills = set(
        extract_skills(job)
    )

    matching = sorted(
        resume_skills & job_skills
    )

    missing = sorted(
        job_skills - resume_skills
    )

    coverage = (
        len(matching)
        / len(job_skills)
        * 100
        if job_skills
        else 0
    )

    tfidf = tfidf_score(
        resume,
        job
    )

    keyword = keyword_score(
        resume,
        job
    )

    final = (
        coverage * 0.50
        + tfidf * 0.30
        + keyword * 0.20
    )

    return {
        "final": round(final, 2),
        "tfidf": round(tfidf, 2),
        "keyword": round(keyword, 2),
        "coverage": round(coverage, 2),
        "matching": matching,
        "missing": missing
    }


HTML = """
<!DOCTYPE html>

<html>

<head>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<title>AI Resume Job Matcher</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #0b1020;
    color: white;
}

.container {
    width: min(1000px, 94%);
    margin: auto;
    padding: 30px 0;
}

.hero {
    text-align: center;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: clamp(30px, 7vw, 55px);
    margin-bottom: 10px;
}

.hero p {
    color: #aab3c5;
}

.card {
    background: #121a2e;
    border: 1px solid #293551;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
}

label {
    display: block;
    margin-bottom: 8px;
    font-weight: bold;
}

textarea {
    width: 100%;
    min-height: 220px;
    resize: vertical;
    border: 1px solid #3b4865;
    background: #0c1325;
    color: white;
    border-radius: 12px;
    padding: 15px;
    outline: none;
    margin-bottom: 18px;
}

textarea:focus {
    border-color: #7c8cff;
}

input[type=file] {
    width: 100%;
    padding: 15px;
    border: 1px dashed #52617e;
    border-radius: 12px;
    margin-bottom: 18px;
    color: white;
}

button {
    width: 100%;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #8b96ff;
    background: transparent;
    color: white;
    font-size: 16px;
    cursor: pointer;
    transition: .2s;
}

button:hover {
    background: #7c8cff;
    color: #080b16;
    transform: translateY(-2px);
}

button:active {
    transform: scale(.98);
}

.score {
    text-align: center;
    font-size: 60px;
    font-weight: bold;
    margin: 20px;
}

.grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(220px, 1fr));
    gap: 15px;
}

.metric {
    padding: 20px;
    border: 1px solid #303d5a;
    border-radius: 14px;
    text-align: center;
}

.metric strong {
    display: block;
    font-size: 28px;
    margin-top: 8px;
}

.skill {
    display: inline-block;
    border: 1px solid #506080;
    padding: 7px 10px;
    border-radius: 20px;
    margin: 4px;
    font-size: 14px;
}

.missing {
    border-color: #a96d6d;
}

.matching {
    border-color: #6b9d7c;
}

</style>

</head>


<body>

<div class="container">

<div class="hero">

<h1>AI Resume–Job Matcher</h1>

<p>
NLP-powered resume analysis and skill-gap detection
</p>

</div>


<div class="card">

<form
action="/analyze"
method="post"
enctype="multipart/form-data">

<label>
Upload Resume PDF
</label>

<input
type="file"
name="resume_file"
accept=".pdf">


<label>
Or paste your resume
</label>

<textarea
name="resume_text"
placeholder="Paste your resume here..."></textarea>


<label>
Job Description
</label>

<textarea
name="job_description"
placeholder="Paste the job description here..."
required></textarea>


<button type="submit">
Analyze Resume
</button>

</form>

</div>

</div>

</body>

</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():

    return HTML


@app.post(
    "/analyze",
    response_class=HTMLResponse
)
async def analyze_resume(

    resume_file: UploadFile = File(None),

    resume_text: str = Form(""),

    job_description: str = Form(...)
):

    if resume_file and resume_file.filename:

        data = await resume_file.read()

        if resume_file.filename.lower().endswith(".pdf"):

            resume_text = extract_pdf_text(data)

    if not resume_text.strip():

        return HTMLResponse(
            """
            <h2 style="font-family:Arial">
            Please provide a resume.
            </h2>
            """,
            status_code=400
        )

    if not job_description.strip():

        return HTMLResponse(
            """
            <h2 style="font-family:Arial">
            Please provide a job description.
            </h2>
            """,
            status_code=400
        )

    result = analyze(
        resume_text,
        job_description
    )

    matching_html = "".join(
        f'<span class="skill matching">✓ {x}</span>'
        for x in result["matching"]
    )

    missing_html = "".join(
        f'<span class="skill missing">✗ {x}</span>'
        for x in result["missing"]
    )

    page = f"""

<!DOCTYPE html>

<html>

<head>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<title>Analysis Result</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #0b1020;
    color: white;
    font-family: Arial, sans-serif;
}}

.container {{
    width: min(950px, 94%);
    margin: auto;
    padding: 30px 0;
}}

.card {{
    background: #121a2e;
    border: 1px solid #293551;
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 20px;
}}

.score {{
    text-align: center;
    font-size: clamp(50px, 12vw, 80px);
    font-weight: bold;
    margin: 20px;
}}

.grid {{
    display: grid;
    grid-template-columns:
    repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
}}

.metric {{
    text-align: center;
    padding: 20px;
    border: 1px solid #303d5a;
    border-radius: 14px;
}}

.metric strong {{
    display: block;
    font-size: 28px;
    margin-top: 8px;
}}

.skill {{
    display: inline-block;
    padding: 8px 12px;
    border: 1px solid #52617e;
    border-radius: 20px;
    margin: 4px;
}}

.missing {{
    border-color: #a96d6d;
}}

.matching {{
    border-color: #6b9d7c;
}}

.back {{
    display: block;
    text-align: center;
    margin-top: 25px;
    color: white;
    text-decoration: none;
    border: 1px solid #7c8cff;
    padding: 13px;
    border-radius: 12px;
}}

.back:hover {{
    background: #7c8cff;
    color: #080b16;
}}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h1>Resume Analysis</h1>

<div class="score">
{result["final"]}%
</div>

<p style="text-align:center">
Overall Match Score
</p>

</div>


<div class="card">

<h2>Matching Metrics</h2>

<div class="grid">

<div class="metric">

TF-IDF Similarity

<strong>
{result["tfidf"]}%
</strong>

</div>


<div class="metric">

Keyword Relevance

<strong>
{result["keyword"]}%
</strong>

</div>


<div class="metric">

Skill Coverage

<strong>
{result["coverage"]}%
</strong>

</div>

</div>

</div>


<div class="card">

<h2>Matching Skills</h2>

{matching_html or "No matching skills detected."}

</div>


<div class="card">

<h2>Missing Skills</h2>

{missing_html or "No major missing skills detected."}

</div>


<a class="back" href="/">
Analyze Another Resume
</a>

</div>

</body>

</html>
"""

    return page
