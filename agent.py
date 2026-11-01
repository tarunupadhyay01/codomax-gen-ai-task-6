"""CareerPilot agent: prompts, tools (including RAG search) and the agent loop.

Workflow:  Understand Task -> Decide Action -> Use Tool -> Process Result -> Respond
"""
import json
import sqlite3
import urllib.parse
import urllib.request
from contextlib import closing
from datetime import datetime
from zoneinfo import ZoneInfo

from google.genai import types

from rag import retrieve, with_retry

MODEL = "gemini-2.5-flash"
MAX_STEPS = 6          # safety limit for tool rounds per request
MAX_HISTORY = 30       # memory limit

# ---------------- Prompt engineering: base prompt + one prompt per mode ----------------
BASE_PROMPT = (
    "You are CareerPilot, an AI career and study assistant with tools. "
    "Workflow: understand the request, decide whether a tool is needed, use tools, check the results, then answer. "
    "Rules: "
    "1) For anything about the user's resume, job description or notes, call search_documents first. "
    "You may call it several times with different queries (for example once for the resume and once for the job description). "
    "Answer only from what it returns and mention the source file in brackets. If nothing relevant is found, say so. "
    "2) Never invent facts about the user or tool results. "
    "3) Use add_task, list_tasks and complete_task to manage the user's study or job-application plan. "
    "4) Use wikipedia_summary for general background facts and get_current_datetime for dates. "
    "5) If a tool returns an error, tell the user or try another approach. "
    "6) Keep answers short, friendly and well organised."
)

MODES = {
    "Career Coach": (
        "Mode: Career Coach. Help with resume improvement, skill gaps, career paths and job applications. "
        "When comparing a resume with a job description, list matching skills, missing skills and 3 concrete improvements."
    ),
    "Interview Coach": (
        "Mode: Interview Coach. Run a mock interview based on the job description and resume (search them first). "
        "Ask ONE question at a time. When the user answers, call score_interview_answer with the question and the answer, "
        "then share the score and feedback and ask the next question. After 5 questions, give a short summary."
    ),
    "Study Helper": (
        "Mode: Study Helper. Explain concepts in simple words using the user's notes (search them first). "
        "Create short quizzes when asked, and turn study plans into tasks with add_task."
    ),
}

EXAMPLES = {
    "Career Coach": [
        "Compare my resume with the job description",
        "What skills am I missing for this job?",
        "Rewrite my project description to sound stronger",
    ],
    "Interview Coach": [
        "Start a mock interview for this job",
        "Give me one tough question about my projects",
    ],
    "Study Helper": [
        "Make a 5-day study plan for the missing skills and add it to my tasks",
        "Explain RAG in simple words",
        "Show my pending tasks",
    ],
}

# ---------------- Tool declarations (what the model sees) ----------------
TOOL_DECLARATIONS = [
    {
        "name": "search_documents",
        "description": "Search the user's uploaded documents (resume, job description, notes) and return the most relevant passages. This is the RAG tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, e.g. 'required skills' or 'work experience'."},
                "top_k": {"type": "integer", "description": "Number of passages to return (default 4)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "score_interview_answer",
        "description": "Evaluate the user's answer to an interview question and return a score out of 10 with feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The interview question that was asked."},
                "answer": {"type": "string", "description": "The user's answer."},
            },
            "required": ["question", "answer"],
        },
    },
    {
        "name": "wikipedia_summary",
        "description": "Get a short factual summary of a topic from Wikipedia.",
        "parameters": {
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "Topic, e.g. 'Retrieval-augmented generation'."}},
            "required": ["topic"],
        },
    },
    {
        "name": "get_current_datetime",
        "description": "Get the current date and time in a timezone (default Asia/Kolkata).",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "IANA timezone, e.g. 'Asia/Kolkata'."}},
        },
    },
    {
        "name": "add_task",
        "description": "Add a task to the user's study or job-application plan (database).",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Task title."},
                "due": {"type": "string", "description": "Optional due date, e.g. '2026-11-10'."},
            },
            "required": ["title"],
        },
    },
    {
        "name": "list_tasks",
        "description": "List tasks from the plan database.",
        "parameters": {
            "type": "object",
            "properties": {"status": {"type": "string", "description": "'pending' (default), 'done' or 'all'."}},
        },
    },
    {
        "name": "complete_task",
        "description": "Mark a task as done using its id number.",
        "parameters": {
            "type": "object",
            "properties": {"task_id": {"type": "integer", "description": "The task id."}},
            "required": ["task_id"],
        },
    },
]


def _get_json(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "codomax-intern-agent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------- Tools ----------------
class Tools:
    """All tools the agent can use. Each user session gets its own Tools object and database."""

    ALLOWED = {"search_documents", "score_interview_answer", "wikipedia_summary",
               "get_current_datetime", "add_task", "list_tasks", "complete_task"}

    def __init__(self, client, db_path):
        self.client = client
        self.store = None          # vector database, set by the app after documents are processed
        self.db_path = db_path
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS tasks ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, "
                "due TEXT, done INTEGER DEFAULT 0)"
            )

    def run(self, name, args):
        """Run a tool by name. Errors are returned as text so the agent can recover."""
        if name not in self.ALLOWED:
            return f"Error: unknown tool '{name}'."
        try:
            return str(getattr(self, name)(**args))
        except Exception as e:  # noqa: BLE001
            return f"Tool error: {e}"

    # ----- RAG tool -----
    def search_documents(self, query, top_k=4):
        if self.store is None:
            return "No documents uploaded yet. Ask the user to upload and process their documents first."
        k = max(1, min(int(float(top_k or 4)), 8))
        hits = retrieve(self.client, self.store, query, k)
        return "\n\n".join(f"[{c['source']}, page {c['page']}] (score {s:.2f})\n{c['text']}" for c, s in hits)

    # ----- Prompt engineering + structured output tool -----
    def score_interview_answer(self, question, answer):
        prompt = (
            "You are a strict but kind interviewer. Evaluate the candidate's answer.\n"
            f"Question: {question}\nAnswer: {answer}\n\n"
            "Return ONLY valid JSON with keys: score (integer 1-10), strengths (list of up to 3 short strings), "
            "improvements (list of up to 3 short strings), better_answer_tip (one short sentence)."
        )
        config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
        response = with_retry(lambda: self.client.models.generate_content(
            model=MODEL, contents=prompt, config=config))
        data = json.loads(response.text)
        score = max(1, min(10, int(data["score"])))
        return (
            f"Score: {score}/10\n"
            f"Strengths: {'; '.join(data.get('strengths', []))}\n"
            f"Improvements: {'; '.join(data.get('improvements', []))}\n"
            f"Tip: {data.get('better_answer_tip', '')}"
        )

    # ----- External API tool -----
    def wikipedia_summary(self, topic):
        def summary(title):
            return _get_json("https://en.wikipedia.org/api/rest_v1/page/summary/"
                             + urllib.parse.quote(title.replace(" ", "_")))
        try:
            data = summary(topic)
        except Exception:  # noqa: BLE001  (not found: search for the closest title)
            found = _get_json("https://en.wikipedia.org/w/api.php?action=opensearch&limit=1&format=json&search="
                              + urllib.parse.quote(topic))
            if not found[1]:
                return f"No Wikipedia page found for '{topic}'."
            data = summary(found[1][0])
        return f"{data.get('title')}: {data.get('extract', '')[:700]}"

    def get_current_datetime(self, timezone="Asia/Kolkata"):
        tz = timezone or "Asia/Kolkata"
        return datetime.now(ZoneInfo(tz)).strftime("%A, %d %B %Y, %I:%M %p ") + tz

    # ----- Database tools -----
    def add_task(self, title, due=""):
        title = title.strip()
        if not title or len(title) > 200:
            raise ValueError("Task title must be 1-200 characters.")
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            cur = conn.execute("INSERT INTO tasks (title, due) VALUES (?, ?)", (title, due or None))
            return f"Task added with id {cur.lastrowid}: {title}" + (f" (due {due})" if due else "")

    def list_tasks(self, status="pending"):
        rows = self.task_rows(status)
        if not rows:
            return "No tasks found."
        return "\n".join(
            f"{i}. {t}" + (f" (due {d})" if d else "") + (" [done]" if done else "")
            for i, t, d, done in rows
        )

    def task_rows(self, status="all"):
        where = {"pending": "WHERE done = 0", "done": "WHERE done = 1", "all": ""}.get(status or "pending", "WHERE done = 0")
        with closing(sqlite3.connect(self.db_path)) as conn:
            return conn.execute(f"SELECT id, title, due, done FROM tasks {where} ORDER BY id").fetchall()

    def complete_task(self, task_id):
        tid = int(float(task_id))
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            cur = conn.execute("UPDATE tasks SET done = 1 WHERE id = ?", (tid,))
            return f"Task {tid} marked as done." if cur.rowcount else f"No task with id {tid}."


# ---------------- Agent memory and loop ----------------
def trim_history(contents, max_items=MAX_HISTORY):
    """Memory limit: keep recent items, starting at a real user message."""
    if len(contents) <= max_items:
        return contents
    for i in range(len(contents) - max_items, len(contents)):
        c = contents[i]
        if c.role == "user" and any(getattr(p, "text", None) for p in c.parts):
            return contents[i:]
    return contents


def run_agent(client, tools, contents, user_text, mode="Career Coach", model=MODEL):
    """Run one user turn. Returns (answer, steps, updated_contents)."""
    contents = trim_history(contents) + [
        types.Content(role="user", parts=[types.Part(text=user_text)])
    ]
    config = types.GenerateContentConfig(
        system_instruction=BASE_PROMPT + " " + MODES.get(mode, ""),
        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.3,
    )
    steps = []
    for _ in range(MAX_STEPS):
        response = with_retry(lambda: client.models.generate_content(
            model=model, contents=contents, config=config))              # Understand + Decide
        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            return "I could not produce a response. Please rephrase your request.", steps, contents
        contents.append(response.candidates[0].content)

        calls = response.function_calls
        if not calls:                                                    # Respond
            return response.text or "Done.", steps, contents

        result_parts = []
        for call in calls:                                               # Use Tool
            args = dict(call.args or {})
            result = tools.run(call.name, args)                          # Process Result
            steps.append({"tool": call.name, "args": args, "result": result})
            result_parts.append(types.Part.from_function_response(
                name=call.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=result_parts))

    return "I reached my step limit before finishing. Please try a simpler request.", steps, contents
