# Holy Grail AI System

**Created by Dakota Rain Lock**

Welcome to the Holy Grail AI System, a proof-of-concept autonomous AI development system. This project is a demonstration of a self-improving AI capable of generating, evolving, and deploying complete web applications from the ground up. It features a sophisticated long-term memory system, a multi-agent architecture, and the ability to learn from its own creations and the web.

This project was built to push the boundaries of what is possible with today's AI and to showcase the skills and passion I bring to the field of software engineering and artificial intelligence.

---

## 🚀 Core Features

The Holy Grail AI System is more than just a code generator; it's an end-to-end autonomous development pipeline.

### 1. **Autonomous Code Generation & Evolution**
*   **Idea Generation:** The system can autonomously conceive of novel web application ideas, leveraging its extensive memory of past projects and web-crawled data.
*   **Code Generation:** It writes complete, functional code for single-page applications, including HTML, Tailwind CSS, and JavaScript.
*   **Iterative Evolution:** Holy Grail evaluates its own generated code against a quality threshold. It then enters an "evolutionary loop," iteratively refining and improving the code, adding new features, and enhancing UI/UX until the quality standard is met.

### 2. **True Long-Term Memory & Learning**
*   **Persistent Memory:** The system records every interaction, project, debug session, and piece of learned information into a persistent JSON-based memory file.
*   **Semantic Vector Cache:** At the heart of its long-term memory is a custom vector cache. This allows the system to perform semantic searches on its memory, retrieving information based on conceptual relevance rather than just keyword matches. This is the key to its "true" long-term memory, as it allows the AI to make connections and draw insights from its entire history.
*   **Closed-Loop Learning:** The system is capable of learning from its own deployed applications. It can extract information and user feedback from the live apps it creates, store this data, and use the resulting insights to inform and improve future projects.

### 3. **Multi-Agent Architecture**
Holy Grail employs a modular, multi-agent architecture where specialized AI agents collaborate to perform complex tasks:
*   **The Emissary:** The user-facing conversational AI that acts as the primary interface to the system.
*   **Memento:** The memory guardian, responsible for retrieving and providing context from the system's vast history.
*   **Dr. Debug:** An expert AI coding assistant that can analyze code, identify bugs, suggest improvements, and perform complex rewrites.
*   **B.E.N.N.I. (Browser-Enabled Neural Navigation Interface):** An AI agent that can navigate a built-in browser, interact with web pages, and extract information in real-time.

### 4. **Web Intelligence with GrailCrawler**
*   To stay current, the system uses **GrailCrawler**, a powerful web-crawling engine that autonomously gathers information from a curated list of high-quality sources, including tech news sites, development blogs, and research repositories like arXiv.
*   This crawled data is processed, vectorized, and integrated into the system's memory, providing a continuous stream of fresh knowledge to inform its development and decision-making processes.

### 5. **Live Deployment Pipeline**
*   The system can deploy its creations directly to **Netlify** via their API, taking a project from an idea to a live, publicly accessible URL in a single, autonomous run.

---

## 🏗️ System Architecture

The system is built around a Python **Flask** backend that orchestrates the various AI agents and services.

```
+--------------------------------+
|      Frontend (index.html)     |
| (HTML, Tailwind CSS, JS)       |
+--------------------------------+
             |
             v
+--------------------------------+
|      Flask Backend API         |
|      (app_backend.py)          |
+--------------------------------+
             |
             v
+--------------------------------+       +-------------------------+
|     AI Core / Multi-Agent      |------>|      Gemini Pro API     |
| (Emissary, Memento, Dr. Debug) |       +-------------------------+
+--------------------------------+
             |
             v
+--------------------------------+
|      Persistent Memory         |
| (context_memory.json)          |
| + Vector Cache & GrailCrawler  |
+--------------------------------+
             |
             v
+--------------------------------+       +-------------------------+
|     Autonomous Pipelines       |------>|      Netlify API        |
| (Generation, Evolution)        |       +-------------------------+
+--------------------------------+

```

---

## 🛠️ Technical Stack

*   **Backend:** Python 3, Flask
*   **AI Model:** Google Gemini API
*   **Web Automation:** Playwright, aiohttp, BeautifulSoup, Trafilatura
*   **Frontend:** HTML, Tailwind CSS, JavaScript
*   **Deployment:** Netlify API

---

## ⚙️ Setup and Installation

To get the Holy Grail AI System running on your local machine, follow these steps:

**1. Prerequisites:**
*   Python 3.10 or higher
*   `pip` for package management
*   Remember to change filepaths to match your own (ctrl + f \mnt\)

**2. Clone the Repository:**
```bash
git clone <repository-url>
cd <repository-name>
```

**3. Set up a Virtual Environment:**
```bash
# For Linux/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

**4. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**5. Configure Environment Variables:**
*   In the `holygrail-opensource` directory, rename `.env.example` to `.env`.
*   Fill in the required API keys and tokens:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    NETLIFY_AUTH_TOKEN="YOUR_NETLIFY_AUTH_TOKEN"
    GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
    GITHUB_USERNAME="YOUR_GITHUB_USERNAME"
    ```

**6. Run the Backend Server:**
```bash
python app_backend.py
```
The Flask server will start, and the system will be running on `http://localhost:5000`.

**7. Launch the Frontend:**
*   Open your web browser and navigate to `http://localhost:5000`.

---

## 👨‍💻 A Note from the Author, Dakota Rain Lock

Thank you for exploring the Holy Grail AI System. This project is the culmination of countless hours of research, development, and a relentless passion for pushing the boundaries of artificial intelligence. My goal was not just to build another code generator, but to create a system that begins to model the processes of learning, creativity, and self-improvement that are hallmarks of true intelligence.

Building Holy Grail has been an incredible journey. It has sharpened my skills in backend development with Python and Flask, deepened my understanding of large language models, and challenged me to design complex, multi-agent systems with persistent memory. Features like the semantic vector cache and the closed-loop learning pipeline are my attempts to solve some of the most exciting problems in AI today.

I am actively seeking opportunities where I can bring this passion, dedication, and technical expertise to a forward-thinking team. I am eager to contribute to projects that are not just technically challenging but also aim to create a meaningful impact.

---

Tips for use from Dakota Rain Lock: 

1. If you launch Holy Grail using a CLI agent (Gemini, Codex, Claude), tell it to run the Flask App nohup, and instruct it to operate the flask app by using curl commands to hit the endpoints, the CLI agent can actually pilot it for hours at a time, enabling true autonomous development.

2. Holy Grail is currently set up to use Gemini 3 Flash as the LLM powering it, but any LLM can run it with minimal changes to the python code.

## 📫 Get in Touch

*   **GitHub:** [github.com/dakotalock](https://github.com/dakotalock)
*   **Email:** [dakota.lock@westernalum.org]
*   **Phone Number:** [7194069402]
