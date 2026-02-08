# Holy Grail Backend Modularization - Sprint Punchlist

**Sprint Goal**: Build a complete, standalone, modular replacement for `app_backend.py` in a new `hg_backend/` package. At sprint end, `python hg_backend/main.py` works identically to `python app_backend.py`.

**Critical Constraint**: `app_backend.py` is NEVER modified. All line references remain stable throughout the sprint. The new `hg_backend/` package extracts and refactors code FROM `app_backend.py` into focused modules.

**Source File**: `app_backend.py` (9,008 lines)
**Target Package**: `hg_backend/`

---

## Dead Code & Duplicates Registry

The following items in `app_backend.py` are superseded by later monkey patches and MUST NOT be carried to the new modules. Only the **final working version** of each function/class gets extracted.

| Original Definition | Lines | Superseded By | Lines | Action |
|---|---|---|---|---|
| `VectorCache` class | L642-764 | `VectorCacheFixed` class + enhanced `find_similar` | L6364-7090 | Extract ONLY the final fixed version |
| `analyze_code_with_debugger()` (1st) | L3996-4131 | `analyze_code_with_debugger()` (2nd, completion forcer) | L8207-8265 | Extract ONLY the completion forcer version |
| `chat_with_memento()` (1st) | L4346-4450 | `enhanced_chat_with_memento()` via monkey patch | L8339-8450 | Extract ONLY the enhanced version |
| `MemoryManager.get_relevant_memory` (original) | inside L2037-2889 | `fixed_get_relevant_memory()` | L8462-8528 | Extract ONLY the fixed version |
| `MemoryManager.analyze` (original) | inside L2037-2889 | `MemoryAnalysisWizard.analyze_comprehensive()` via monkey patch | L8189-8197 | Extract wizard version as the analyze method |
| `GrailCrawler.crawl_latest_data` (original) | inside L1478-2035 | `optimized_crawl()` via `GrailCrawlerUpgrade` + `chunked_crawl_latest_data` | L5206-5233, L8839-8857 | Integrate optimizations into the crawler class directly |
| `VectorCache.find_similar` (original) | L716-743 | `debug_find_similar()` wrapping enhanced semantic version | L8540-8573, L6900-7090 | Merge semantic + debug into one clean implementation |
| `fix_relative_urls()` (1st) | L519-573 | `fix_relative_urls_enhanced()` | L5312-5410 | Extract ONLY the enhanced version |
| Redundant imports at L6356-6362, L7096-7102, L7410-7416 | scattered | Top-level imports at L1-41 | L1-41 | Single import block at top of each module |
| `logging.basicConfig` + `logger` (2nd def) | L5001-5003 | Original at L46-47 | L46-47 | Single logging setup |

---

## Package Directory Structure

```
hg_backend/
  __init__.py
  main.py                        # Entry point (replaces if __name__ == '__main__' block)
  app.py                         # Flask app factory + CORS + blueprint registration
  config.py                      # TurboConfig, Config, ConsoleColors
  logger.py                      # Logging setup + print_error/success/info/warning/debug
  content_extraction.py          # extract_page_content_advanced, extract_with_playwright, etc.
  task_manager.py                # TaskManager class + thread pool + queue
  vector_cache.py                # Unified VectorCache (merged VectorCacheFixed + semantic enhancements)
  memory_retriever.py            # SmartMemoryRetriever class
  token_pruner.py                # TokenPruner class
  closed_loop_context.py         # ClosedLoopLearningContext class
  idea_validator.py              # IdeaValidator class
  live_data_fetcher.py           # LiveDataFetcher class
  grail_crawler.py               # GrailCrawler class (with integrated optimizations)
  memory_manager.py              # MemoryManager class (with fixed get_relevant_memory)
  prompts.py                     # Prompts class (all LLM prompt templates)
  llm_client.py                  # call_gemini_api() with retry + model fallback
  code_generation.py             # generate_frontend_code, add_watermark, evaluate_code_quality
  idea_generation.py             # generate_autonomous_idea
  code_evolution.py              # evolve_app_code
  code_analysis.py               # analyze_code_with_debugger (completion forcer), debug_chat, rewrite_code_section
  deployment.py                  # deploy_to_netlify_direct, robust_cleanup
  browser_intelligence.py        # BrowserIntelligence class
  reddit_extractor.py            # RedditCommentExtractor class
  benni_agent.py                 # generate_benni_response
  closed_loop_learning.py        # ClosedLoopLearning class
  memory_analysis.py             # MemoryAnalysisWizard class
  memory_agents.py               # chat_with_memento (enhanced), _prepare_memento_system_analysis_chunks
  agent_chunker.py               # UniversalAgentChunker class
  routes/
    __init__.py                  # Blueprint registration helper
    browser.py                   # /browser/navigate, /proxy, /proxy-form, /browser/session
    memory.py                    # /memory, /memory/analyze, /debug/memory
    chat.py                      # /chat, /debug/chat, /api/v1/memento-chat-working
    code_ops.py                  # /debug/analyze, /debug/rewrite
    projects.py                  # /generate-and-deploy, /evolve-app
    data.py                      # /live-data, /tasks/<task_id>, /models, /rlhf/feedback
    static.py                    # /, /holygrail, /holygrail/<filename>, prism placeholders
```

---

## Task List

### Task 1: Create Package Directory Structure
**Status**: [ ] Not Started
**Parallel Group**: Sequential (must be first)
**Source Lines**: N/A
**Output File(s)**: `hg_backend/`, `hg_backend/__init__.py`, `hg_backend/routes/`, `hg_backend/routes/__init__.py`
**Description**: Create the `hg_backend/` package directory and `routes/` subdirectory. Write empty `__init__.py` files as placeholders. The top-level `__init__.py` should contain a docstring and version string. The `routes/__init__.py` should be empty initially (populated in Task 37).
**Acceptance Criteria**:
- `hg_backend/` directory exists with `__init__.py`
- `hg_backend/routes/` directory exists with `__init__.py`
- `python -c "import hg_backend"` succeeds without error
**Dependencies**: None

---

### Task 2: Extract config.py
**Status**: [ ] Not Started
**Parallel Group**: A
**Source Lines**: app_backend.py L10-11 (os, Path imports), L21 (dotenv), L66-67 (load_dotenv), L72-193 (TurboConfig, Config, env validation, ConsoleColors)
**Output File(s)**: `hg_backend/config.py`
**Description**: Extract `TurboConfig` (L73-80), `Config` (L83-181), environment variable validation (L183-192), and `ConsoleColors` (L194-204) into a standalone config module. Call `load_dotenv()` at module level. The `Config.BASE_DIR` should default to a sensible path but allow override via `BASE_DIR` env var (as it already does). Keep the `MODELS` dict, `LIVE_DATA_APIS`, `CRAWLER_*` settings, and `SUPPORTED_TECH` exactly as-is so the frontend contract is preserved.
**Acceptance Criteria**:
- `from hg_backend.config import Config, TurboConfig, ConsoleColors` works
- All Config attributes match original values
- Environment variable validation runs on import and exits if keys are missing
- `python -c "from hg_backend.config import Config; print(Config.GEMINI_API_KEY)"` prints the key (or exits if not set)
**Dependencies**: Task 1

---

### Task 3: Extract logger.py
**Status**: [ ] Not Started
**Parallel Group**: A
**Source Lines**: app_backend.py L24 (logging import), L45-47 (logging setup), L207-222 (print_error, print_success, print_info, print_warning, print_debug)
**Output File(s)**: `hg_backend/logger.py`
**Description**: Extract logging configuration and the five colored print helper functions. Import `ConsoleColors` from `config.py`. Set up `logging.basicConfig` and create the module-level `logger`. The `print_debug` function checks `DEBUG_MODE` env var. Do NOT duplicate `logging.basicConfig` (the second call at L5001-5003 is dead code).
**Acceptance Criteria**:
- `from hg_backend.logger import print_error, print_success, print_info, print_warning, print_debug, logger` works
- Each function produces colored output with correct prefix (ERROR, SUCCESS, INFO, WARNING, DEBUG)
- `print_debug` only outputs when `DEBUG_MODE=true`
**Dependencies**: Task 1, Task 2

---

### Task 4: Extract task_manager.py
**Status**: [ ] Not Started
**Parallel Group**: A
**Source Lines**: app_backend.py L585-641 (request_pool, background_tasks, TaskManager class)
**Output File(s)**: `hg_backend/task_manager.py`
**Description**: Extract the `ThreadPoolExecutor` (`request_pool`), `Queue` (`background_tasks`), and `TaskManager` class. Import `TurboConfig` from config for `MAX_CONCURRENT_REQUESTS` and `MAX_TASK_QUEUE`. The `TaskManager` uses class-level `_tasks` dict and `_task_counter`. Keep `create_task`, `get_task_status`, `update_task`, and `cleanup_old_tasks` methods.
**Acceptance Criteria**:
- `from hg_backend.task_manager import TaskManager, request_pool, background_tasks` works
- `TaskManager.create_task(lambda: "test")` returns a task ID string
- `TaskManager.get_task_status(task_id)` returns status dict
**Dependencies**: Task 1, Task 2

---

### Task 5: Extract token_pruner.py
**Status**: [ ] Not Started
**Parallel Group**: A
**Source Lines**: app_backend.py L912-1244 (TokenPruner class)
**Output File(s)**: `hg_backend/token_pruner.py`
**Description**: Extract the `TokenPruner` class with all its methods: `_count_tokens`, `_score_message`, `_keyword_overlap`, `_select_segments`, `_distill_segment`, `build_payload`, and the `TOKEN_LIMIT` constant. Import `Config` from config for model references if needed. The chunked_build_payload monkey patch (L8796-8807) should be integrated directly into the `build_payload` method -- if content exceeds a threshold, chunk it using inline logic (do NOT import UniversalAgentChunker here to avoid circular deps; instead inline the simple chunking logic from L8599-8629).
**Acceptance Criteria**:
- `from hg_backend.token_pruner import TokenPruner` works
- `TokenPruner.build_payload(system_context="test", prompt_text="hello")` returns a tuple of (contents, selected_tokens)
- Token counting works correctly
**Dependencies**: Task 1, Task 2, Task 3

---

### Task 6: Extract vector_cache.py (Unified)
**Status**: [ ] Not Started
**Parallel Group**: B
**Source Lines**: app_backend.py L642-764 (original VectorCache), L6352-7090 (VectorCacheFixed + enhanced semantic vectorization + intent-based boosting), L7660-7838 (meaningful snippets enhancement), L8535-8573 (debug enhancement)
**Output File(s)**: `hg_backend/vector_cache.py`
**Description**: Create a SINGLE unified `VectorCache` class that merges the best of all versions. The original VectorCache (L642-764) provides the basic structure. VectorCacheFixed (L6364-6600) adds all-data scanning. The enhanced semantic vectorization (L6900-7090) adds intent understanding, content type detection, temporal context analysis, entity recognition, and intent-based boosting. The meaningful snippets patch (L7663-7838) adds `SmartMemoryRetriever` integration. The debug patch (L8535-8573) adds logging.

The unified class MUST have:
- `initialize()` - loads from disk AND scans all system data (from VectorCacheFixed)
- `_text_to_vector()` - the ENHANCED version with intent awareness (from L6900+ semantic vectorization)
- `_cosine_similarity()` - standard implementation
- `add_item()` - thread-safe with lock
- `find_similar()` - uses enhanced semantic matching with intent-based boosting, includes debug logging, returns meaningful snippets
- `_save_cache()` - persist to disk
- `get_stats()` - cache statistics

Do NOT carry over: the original basic `_text_to_vector` (L668-684), the basic `find_similar` (L716-743), or any of the monkey-patching wrapper functions.
**Acceptance Criteria**:
- `from hg_backend.vector_cache import VectorCache` works
- `VectorCache.initialize()` loads cache from disk
- `VectorCache.add_item("test", "some text content")` succeeds
- `VectorCache.find_similar("text content")` returns list of (id, score) tuples
- Semantic enhancement (intent vectors, entity boosting) is integrated
- Debug logging is present
**Dependencies**: Task 1, Task 2, Task 3

---

### Task 7: Extract memory_retriever.py
**Status**: [ ] Not Started
**Parallel Group**: B
**Source Lines**: app_backend.py L766-911 (SmartMemoryRetriever class)
**Output File(s)**: `hg_backend/memory_retriever.py`
**Description**: Extract the `SmartMemoryRetriever` class with its two static methods: `extract_meaningful_snippet()` (L772-842) and `get_most_relevant_crawled_data()` (L844-910). The `get_most_relevant_crawled_data` method references `MemoryManager.load()` -- use a lazy import inside the method body to avoid circular dependency (MemoryManager depends on VectorCache, SmartMemoryRetriever is used by VectorCache and MemoryManager).
**Acceptance Criteria**:
- `from hg_backend.memory_retriever import SmartMemoryRetriever` works
- `SmartMemoryRetriever.extract_meaningful_snippet("long text here...", query="test", max_length=150)` returns a string <= 150 chars
- All 4 extraction strategies are preserved (query-relevant, density, key phrases, unique middle)
**Dependencies**: Task 1, Task 2, Task 3

---

### Task 8: Extract closed_loop_context.py
**Status**: [ ] Not Started
**Parallel Group**: B
**Source Lines**: app_backend.py L1246-1304 (ClosedLoopLearningContext class)
**Output File(s)**: `hg_backend/closed_loop_context.py`
**Description**: Extract the `ClosedLoopLearningContext` class with `load()` and `get_context_block()` methods. Import `Config` and `TurboConfig` from config, and `print_warning` from logger. Thread-safe caching with TTL is preserved.
**Acceptance Criteria**:
- `from hg_backend.closed_loop_context import ClosedLoopLearningContext` works
- `ClosedLoopLearningContext.load()` returns dict (possibly empty)
- `ClosedLoopLearningContext.get_context_block()` returns formatted string
- TTL caching works (second call within TTL returns cached data)
**Dependencies**: Task 1, Task 2, Task 3

---

### Task 9: Extract idea_validator.py
**Status**: [ ] Not Started
**Parallel Group**: B
**Source Lines**: app_backend.py L1305-1393 (IdeaValidator class)
**Output File(s)**: `hg_backend/idea_validator.py`
**Description**: Extract the `IdeaValidator` class with `validate_idea()` and `_jaccard_similarity()` methods. The validate_idea method at L1310-1380 loads/creates the idea validation JSON file, checks similarity against previous ideas, calls the LLM for scoring, and stores validated ideas. Import `Config` from config and use a lazy import for `call_gemini_api` from llm_client to avoid circular deps. Include the fix from the monkey patch at L7350-7401 where idea storage was corrected -- integrate that fix directly into the method.
**Acceptance Criteria**:
- `from hg_backend.idea_validator import IdeaValidator` works
- `IdeaValidator.validate_idea("Build a weather app", "frontend")` returns a dict with score and feedback
- `IdeaValidator._jaccard_similarity("hello world", "hello there")` returns a float between 0 and 1
- Idea storage works correctly (fix from L7350-7401 integrated)
**Dependencies**: Task 1, Task 2, Task 3

---

### Task 10: Extract live_data_fetcher.py
**Status**: [ ] Not Started
**Parallel Group**: B
**Source Lines**: app_backend.py L1394-1476 (LiveDataFetcher class)
**Output File(s)**: `hg_backend/live_data_fetcher.py`
**Description**: Extract the `LiveDataFetcher` class with `_process_api_data()` and `get_all_live_data()` methods. Import `Config`, `TurboConfig` from config and logging helpers from logger. The `get_all_live_data` method calls `MemoryManager.update_live_data()` at L1466 -- use a lazy import for MemoryManager inside the method to avoid circular dependency.
**Acceptance Criteria**:
- `from hg_backend.live_data_fetcher import LiveDataFetcher` works
- `LiveDataFetcher.get_all_live_data()` returns a dict with keys like 'joke', 'news', 'weather', 'tech_news'
- Caching with TTL works (repeated calls within TTL use cached data)
- API data processing works for all 4 data types
**Dependencies**: Task 1, Task 2, Task 3

---

### Task 11: Extract content_extraction.py
**Status**: [ ] Not Started
**Parallel Group**: B
**Source Lines**: app_backend.py L225-518 (extract_page_content_advanced, extract_with_playwright, fetch_with_playwright, handle_reddit_specific_loading), L519-573 (fix_relative_urls -- SUPERSEDED), L5312-5410 (fix_relative_urls_enhanced -- USE THIS ONE), L575-583 (handle_remove_readonly)
**Output File(s)**: `hg_backend/content_extraction.py`
**Description**: Extract all content extraction functions. Use `fix_relative_urls_enhanced` (L5312-5410) as the canonical `fix_relative_urls` function -- do NOT carry over the original simpler version (L519-573). Also extract `handle_remove_readonly` (L575-583). Import logger helpers and Config as needed. These functions use `asyncio`, `playwright`, `trafilatura`, `html2text`, and `BeautifulSoup`.
**Acceptance Criteria**:
- `from hg_backend.content_extraction import extract_page_content_advanced, fix_relative_urls, handle_remove_readonly` works
- `fix_relative_urls` is the enhanced version (handles mailto:, tel:, has try/except per URL)
- All extraction methods (trafilatura, html2text, BeautifulSoup, playwright) are available
**Dependencies**: Task 1, Task 2, Task 3

---

### Task 12: Extract prompts.py
**Status**: [ ] Not Started
**Parallel Group**: C
**Source Lines**: app_backend.py L2890-3366 (Prompts class)
**Output File(s)**: `hg_backend/prompts.py`
**Description**: Extract the entire `Prompts` class with all prompt templates: `BASE_CONTEXT_TEMPLATE`, `DR_DEBUG_ANALYZE_PROMPT`, `DR_DEBUG_REWRITE_PROMPT`, `GENERATE_FRONTEND_CODE_PROMPT`, `GENERATE_GAME_CODE_PROMPT`, `UNRESTRICTED_FRONTEND_PROMPT`, `EMISSARY_SYSTEM_PROMPT`, `MEMENTO_CHAT_SYSTEM_PROMPT`, `DR_DEBUG_SYSTEM_PROMPT`, `AUTONOMOUS_FRONTEND_IDEA_PROMPT`, `EVALUATE_FRONTEND_CODE_PROMPT`, `EVALUATE_BACKEND_CODE_PROMPT`, `EVOLUTION_PROMPT`, `EVOLUTION_WITH_GOAL_PROMPT`, `MEMORY_ANALYSIS_PROMPT`. This is a pure data class with no dependencies on other modules (just string templates with format placeholders).
**Acceptance Criteria**:
- `from hg_backend.prompts import Prompts` works
- All prompt templates are accessible as class attributes
- `Prompts.BASE_CONTEXT_TEMPLATE.format(...)` works with the expected keyword arguments
- No external imports required (pure string data)
**Dependencies**: Task 1

---

### Task 13: Extract llm_client.py
**Status**: [ ] Not Started
**Parallel Group**: C
**Source Lines**: app_backend.py L3367-3450 (call_gemini_api function)
**Output File(s)**: `hg_backend/llm_client.py`
**Description**: Extract `call_gemini_api()` with its retry logic, model fallback chain, and error handling. Import `Config` from config, `TokenPruner` from token_pruner, and logger helpers. The function builds payloads via `TokenPruner.build_payload()`, makes HTTP requests to the Gemini API, handles 429/503 retries with exponential backoff, and falls back through multiple models. Keep the exact same function signature: `call_gemini_api(model_name, prompt_text=None, conversation_history=None, temperature=0.7, system_context=None)`.
**Acceptance Criteria**:
- `from hg_backend.llm_client import call_gemini_api` works
- Function signature matches original
- Retry logic with 3 attempts and model fallback is preserved
- Returns string response from Gemini API
**Dependencies**: Task 1, Task 2, Task 3, Task 5

---

### Task 14: Extract grail_crawler.py
**Status**: [ ] Not Started
**Parallel Group**: C
**Source Lines**: app_backend.py L1478-2035 (GrailCrawler class), L5189-5236 (GrailCrawlerUpgrade optimizations)
**Output File(s)**: `hg_backend/grail_crawler.py`
**Description**: Extract the `GrailCrawler` class (approximately 560 lines). This is the largest single class. It contains `NEWS_FEEDS` dict (8 categories of RSS feeds), `crawl_latest_data()`, `_crawl_single_feed()`, `_extract_from_rss()`, `_extract_content()`, `_check_robots_txt()`, `_rate_limit()`, `start_autonomous_crawl()`, `stop_autonomous_crawl()`, `_autonomous_crawl_loop()`, `_extract_entities()`, and more.

Integrate the `GrailCrawlerUpgrade` optimizations (L5206-5233) directly into `crawl_latest_data()` rather than as a monkey patch -- after crawling, prioritize high-value domains and sort results. Also integrate the chunked crawl enhancement (L8839-8857) for chunking large `full_text` fields.

Import `Config`, `TurboConfig`, logger helpers, `content_extraction` functions, and `VectorCache` as needed. Use lazy import for `MemoryManager` where the crawler updates memory.
**Acceptance Criteria**:
- `from hg_backend.grail_crawler import GrailCrawler` works
- `GrailCrawler.crawl_latest_data()` returns list of dicts with required shape: `{source, title, description, snippet, full_text, timestamp, category, entities}`
- High-value domain prioritization is built-in (not monkey-patched)
- Large content chunking is built-in
- RSS/Atom parsing works for all 8 feed categories
- Rate limiting and robots.txt compliance preserved
**Dependencies**: Task 1, Task 2, Task 3, Task 6, Task 11

---

### Task 15: Extract memory_manager.py
**Status**: [ ] Not Started
**Parallel Group**: Sequential
**Source Lines**: app_backend.py L2037-2889 (MemoryManager class), L8462-8528 (fixed_get_relevant_memory), L8189-8197 (wizard analyze patch)
**Output File(s)**: `hg_backend/memory_manager.py`
**Description**: Extract the `MemoryManager` class (approximately 850 lines). This is the second-largest class. Key methods include: `_get_base_memory_structure()`, `initialize()`, `load()`, `save()`, `_heal_schema()`, `add_interaction()`, `add_project()`, `add_debug_session()`, `add_full_stack_project()`, `update_browser_session()`, `update_browser_history()`, `add_benni_interaction()`, `add_rlhf_feedback()`, `update_live_data()`, `update_crawled_data()`, `get_relevant_memory()`, `_enhance_memory_item()`, `update_tech_stats()`, `analyze()`.

CRITICAL: Replace the original `get_relevant_memory` with the fixed version from L8462-8528 that properly handles both tuple and dict formats from VectorCache. Replace `analyze()` with a call to `MemoryAnalysisWizard.analyze_comprehensive()` (the wizard replaces the analyze method via monkey patch at L8189-8197 -- integrate this directly).

Import `Config`, `TurboConfig`, logger helpers, `VectorCache`, `SmartMemoryRetriever`. Use lazy import for `MemoryAnalysisWizard` in the analyze method to avoid circular dependency.
**Acceptance Criteria**:
- `from hg_backend.memory_manager import MemoryManager` works
- `MemoryManager.initialize()` creates/loads memory file
- `MemoryManager.load()` returns dict with all required keys
- `MemoryManager.save(data)` persists to disk
- `MemoryManager.get_relevant_memory(query)` uses the FIXED version (handles tuple and dict vector results)
- `MemoryManager.analyze()` delegates to MemoryAnalysisWizard
- Schema healing works on corrupted/outdated memory files
- All add_* methods work correctly
**Dependencies**: Task 1, Task 2, Task 3, Task 6, Task 7

---

### Task 16: Extract code_generation.py
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L3451-3722 (add_watermark, evaluate_code_quality, generate_frontend_code with code parsing, CORS proxy handling, game file generation, error page generation)
**Output File(s)**: `hg_backend/code_generation.py`
**Description**: Extract `add_watermark()` (injects HTML watermark), `evaluate_code_quality()` (LLM-based code scoring), and `generate_frontend_code()` (main frontend generation function, ~220 lines). The `generate_frontend_code` function is complex: it builds context from memory, calls Gemini for code generation, parses HTML/JS/CSS from the response, adds watermarks, handles CORS proxying, and generates game project files. Import `Config`, logger helpers, `call_gemini_api`, `Prompts`, `MemoryManager`, `SmartMemoryRetriever`, `VectorCache`, and `ClosedLoopLearningContext`.
**Acceptance Criteria**:
- `from hg_backend.code_generation import generate_frontend_code, add_watermark, evaluate_code_quality` works
- `add_watermark("<html><body>test</body></html>")` injects the Holy Grail watermark comment
- `generate_frontend_code(concept, stack_type, model_name)` returns HTML string
- Code block parsing (extracting HTML from ```html ``` blocks) works
- Game project multi-file generation works
**Dependencies**: Task 1, Task 2, Task 3, Task 5, Task 6, Task 7, Task 8, Task 12, Task 13, Task 15

---

### Task 17: Extract idea_generation.py
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L3724-3830 (generate_autonomous_idea function)
**Output File(s)**: `hg_backend/idea_generation.py`
**Description**: Extract `generate_autonomous_idea()` (~110 lines). This function assembles deep context from memory (projects, live data, crawled data, agent activity, tech stats, closed loop learning), then calls Gemini to generate an application idea, and validates it via `IdeaValidator`. Import `MemoryManager`, `LiveDataFetcher`, `SmartMemoryRetriever`, `ClosedLoopLearningContext`, `IdeaValidator`, `call_gemini_api`, `Prompts`, and logger helpers.
**Acceptance Criteria**:
- `from hg_backend.idea_generation import generate_autonomous_idea` works
- `generate_autonomous_idea("frontend")` returns idea string
- Context assembly includes all 6 components (system analysis, memory, live data, crawled data, agent activity, tech stats)
- Idea validation is performed before returning
**Dependencies**: Task 1, Task 2, Task 3, Task 8, Task 9, Task 10, Task 12, Task 13, Task 15

---

### Task 18: Extract code_evolution.py
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L3832-3994 (evolve_app_code function)
**Output File(s)**: `hg_backend/code_evolution.py`
**Description**: Extract `evolve_app_code()` (~160 lines). This function iteratively improves code using memory context, live data, crawled data, and evolution prompts. Import `MemoryManager`, `SmartMemoryRetriever`, `LiveDataFetcher`, `call_gemini_api`, `Prompts`, `add_watermark`, and logger helpers.
**Acceptance Criteria**:
- `from hg_backend.code_evolution import evolve_app_code` works
- `evolve_app_code(code, iteration=1, code_type="frontend")` returns tuple of (evolved_code, feedback_message)
- Context assembly is concise (uses cached data)
- Code block parsing works for extracting evolved code from LLM response
**Dependencies**: Task 1, Task 2, Task 3, Task 7, Task 10, Task 12, Task 13, Task 15, Task 16

---

### Task 19: Extract code_analysis.py
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L8207-8265 (analyze_code_with_debugger -- FINAL version, completion forcer), L4133-4250 (debug_chat), L4253-4310 (rewrite_code_section)
**Output File(s)**: `hg_backend/code_analysis.py`
**Description**: Extract three functions:
1. `analyze_code_with_debugger()` -- use the FINAL completion forcer version from L8207-8265 (NOT the earlier version at L3996-4131). This version forces all 10 sections and retries if incomplete.
2. `debug_chat()` from L4133-4250 -- multi-turn debugging conversation with Dr. Debug.
3. `rewrite_code_section()` from L4253-4310 -- code rewriting with instructions.

Import `MemoryManager`, `SmartMemoryRetriever`, `VectorCache`, `call_gemini_api`, `Prompts`, `add_watermark`, and logger helpers.
**Acceptance Criteria**:
- `from hg_backend.code_analysis import analyze_code_with_debugger, debug_chat, rewrite_code_section` works
- `analyze_code_with_debugger(code)` returns analysis string with 10 sections
- Completion forcing (retry on missing sections) is present
- `debug_chat(conversation_history)` returns reply string
- `rewrite_code_section(code, instructions)` returns rewritten code string
**Dependencies**: Task 1, Task 2, Task 3, Task 6, Task 7, Task 12, Task 13, Task 15

---

### Task 20: Extract deployment.py
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L4452-4544 (deploy_to_netlify_direct), L4546-4575 (robust_cleanup)
**Output File(s)**: `hg_backend/deployment.py`
**Description**: Extract `deploy_to_netlify_direct()` (~90 lines) and `robust_cleanup()` (~30 lines). The deployment function handles Netlify site creation, ZIP upload, and deploy status polling. Import `Config` and logger helpers. Uses `requests`, `shutil`, `os`, `Path`, `time`.
**Acceptance Criteria**:
- `from hg_backend.deployment import deploy_to_netlify_direct, robust_cleanup` works
- Function signatures match originals
- ZIP creation, upload, and status polling logic preserved
- `robust_cleanup(path)` handles read-only files with retries
**Dependencies**: Task 1, Task 2, Task 3

---

### Task 21: Extract browser_intelligence.py
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L4790-4999 (BrowserIntelligence class)
**Output File(s)**: `hg_backend/browser_intelligence.py`
**Description**: Extract the `BrowserIntelligence` class with methods: `initialize()`, `get_optimized_headers()`, `_get_domain_strategy()`, `detect_challenge_page()`, `enhance_content()`. The class provides AI-powered browser enhancement with anti-bot bypass strategies for Cloudflare, Reddit, Twitter, etc. Import `ConsoleColors` from config and logger helpers.
**Acceptance Criteria**:
- `from hg_backend.browser_intelligence import BrowserIntelligence` works
- `BrowserIntelligence.initialize()` sets up the intelligence system
- `BrowserIntelligence.get_optimized_headers(url)` returns dict of headers
- `BrowserIntelligence.detect_challenge_page(html)` returns bool
- Domain-specific strategies for Cloudflare, Reddit, Twitter are preserved
**Dependencies**: Task 1, Task 2, Task 3

---

### Task 22: Extract reddit_extractor.py
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L5005-5187 (RedditCommentExtractor class)
**Output File(s)**: `hg_backend/reddit_extractor.py`
**Description**: Extract the `RedditCommentExtractor` class with async methods: `extract_reddit_comments()` and `inject_comments_into_html()`. Uses Playwright for headless browser extraction of Reddit comments with multiple selector strategies. Import logger helpers.
**Acceptance Criteria**:
- `from hg_backend.reddit_extractor import RedditCommentExtractor` works
- `RedditCommentExtractor.extract_reddit_comments(url)` is an async method
- Multiple comment selector strategies are preserved
- HTML injection for comment display works
**Dependencies**: Task 1, Task 2, Task 3

---

### Task 23: Extract benni_agent.py
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L4577-4785 (generate_benni_response function)
**Output File(s)**: `hg_backend/benni_agent.py`
**Description**: Extract `generate_benni_response()` (~220 lines). This function generates BENNI assistant responses by doing dual extraction (Holy Grail UI + browser content), integrating memory storage, performing GrailCrawler-level extraction, and adding to vector cache. Import `MemoryManager`, `VectorCache`, `SmartMemoryRetriever`, `call_gemini_api`, `Prompts`, `extract_page_content_advanced`, and logger helpers. Include the enhanced BENNI from the closed loop learning patch (L7620-7650) that stores extracted data for learning.
**Acceptance Criteria**:
- `from hg_backend.benni_agent import generate_benni_response` works
- Function handles page content extraction and AI response generation
- Memory storage integration works
- Vector cache addition works
- Closed loop learning data storage is integrated (not monkey-patched)
**Dependencies**: Task 1, Task 2, Task 3, Task 6, Task 7, Task 11, Task 12, Task 13, Task 15

---

### Task 24: Extract closed_loop_learning.py
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L7407-7653 (ClosedLoopLearning class + enhanced BENNI + enhanced project generation + enhanced deployment), L8270-8327 (vector compatibility patch)
**Output File(s)**: `hg_backend/closed_loop_learning.py`
**Description**: Extract the `ClosedLoopLearning` class with methods: `initialize()`, `store_extraction()`, `record_learning_cycle()`, `get_improvement_insights()`, `_save_learning_data()`. Integrate the vector compatibility patch from L8274-8326 directly into `get_improvement_insights()` (handle both tuple and dict results from VectorCache). Import `Config`, `VectorCache`, and logger helpers.
**Acceptance Criteria**:
- `from hg_backend.closed_loop_learning import ClosedLoopLearning` works
- `ClosedLoopLearning.initialize()` loads learning data from disk
- `ClosedLoopLearning.store_extraction(url, data, extraction_type)` stores data
- `ClosedLoopLearning.get_improvement_insights(project_type, concept)` returns list of insights
- Vector cache compatibility (tuple and dict) is built-in
**Dependencies**: Task 1, Task 2, Task 3, Task 6

---

### Task 25: Extract memory_analysis.py
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L7843-8197 (MemoryAnalysisWizard class + monkey patch application)
**Output File(s)**: `hg_backend/memory_analysis.py`
**Description**: Extract the `MemoryAnalysisWizard` class with methods: `analyze_comprehensive()`, `_build_streaming_context()`, `_stream_process_context()`, `_call_analysis_with_retry()`, `_incremental_comprehensive_analysis()`, `_analyze_projects_stage()`, `_analyze_debug_stage()`, `_analyze_patterns_stage()`, `_synthesize_analysis()`. Import `call_gemini_api`, `Config`, `TurboConfig`, and logger helpers. Use lazy import for `MemoryManager` (circular dep: MemoryManager.analyze delegates to this wizard).
**Acceptance Criteria**:
- `from hg_backend.memory_analysis import MemoryAnalysisWizard` works
- `MemoryAnalysisWizard.analyze_comprehensive()` runs multi-stage analysis
- Streaming context building processes data incrementally without crashing
- Retry logic with exponential backoff works
- All 4 analysis stages (projects, debug, patterns, synthesis) function
**Dependencies**: Task 1, Task 2, Task 3, Task 13

---

### Task 26: Extract memory_agents.py
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L4312-4343 (_prepare_memento_system_analysis_chunks), L8333-8450 (enhanced chat_with_memento -- FINAL version), L7092-7401 (get_holy_grail_source_context + agent enhancements)
**Output File(s)**: `hg_backend/memory_agents.py`
**Description**: Extract agent-related functions:
1. `_prepare_memento_system_analysis_chunks()` (L4312-4343)
2. `chat_with_memento()` -- use the FINAL enhanced version from L8339-8450 (NOT the original at L4346-4450). This version properly uses vector cache with meaningful snippets, includes crawled data context, agent activity tracking, and full Holy Grail source context.
3. `get_holy_grail_source_context()` (L7104-7200) -- generates source code summary for agents.

Import `MemoryManager`, `VectorCache`, `SmartMemoryRetriever`, `LiveDataFetcher`, `call_gemini_api`, `Prompts`, and logger helpers.

Note: The `HOLY_GRAIL_SOURCE_CONTEXT` global variable (set around L7200) should be computed lazily or on first call rather than at import time.
**Acceptance Criteria**:
- `from hg_backend.memory_agents import chat_with_memento, get_holy_grail_source_context` works
- `chat_with_memento(conversation_history)` uses vector cache with meaningful snippets
- Holy Grail source context generation works (reads and summarizes source files)
- Memento system analysis chunking works
**Dependencies**: Task 1, Task 2, Task 3, Task 6, Task 7, Task 10, Task 12, Task 13, Task 15

---

### Task 27: Extract agent_chunker.py
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L8592-8962 (UniversalAgentChunker class + prepare_agent_context patch + universal agent wrapper)
**Output File(s)**: `hg_backend/agent_chunker.py`
**Description**: Extract the `UniversalAgentChunker` class with methods: `chunk_content_for_agent()`, `prepare_pipeline_context()`, and `prepare_agent_context()` (integrate the monkey patch from L8941-8960 directly into the class). Also extract `create_universal_agent_wrapper()` (L8861-8923) as a standalone function. Import logger helpers only.
**Acceptance Criteria**:
- `from hg_backend.agent_chunker import UniversalAgentChunker, create_universal_agent_wrapper` works
- `UniversalAgentChunker.chunk_content_for_agent("long text", "TestAgent", 50)` returns list of string chunks
- `UniversalAgentChunker.prepare_pipeline_context(context_data, "TestPipeline")` returns string
- `UniversalAgentChunker.prepare_agent_context("TestAgent", context_data)` works (no monkey patch needed)
**Dependencies**: Task 1, Task 2, Task 3

---

### Task 28: Extract routes/static.py
**Status**: [ ] Not Started
**Parallel Group**: E
**Source Lines**: app_backend.py L6200-6250 (approximate -- /, /holygrail, /holygrail/<filename>, prism placeholder routes)
**Output File(s)**: `hg_backend/routes/static.py`
**Description**: Extract static file serving routes into a Flask Blueprint named `static_bp`. Routes: `/` and `/holygrail` (serve index.html from holygrail directory), `/holygrail/<filename>` (serve any file from holygrail directory), `/prism-python-javascript` and `/prism-python.css` (placeholder routes). Import `send_from_directory` from Flask and `Config` for the base directory path.
**Acceptance Criteria**:
- `from hg_backend.routes.static import static_bp` works
- Blueprint has routes for `/`, `/holygrail`, `/holygrail/<filename>`
- Static file serving works from the correct directory
**Dependencies**: Task 1, Task 2

---

### Task 29: Extract routes/data.py
**Status**: [ ] Not Started
**Parallel Group**: E
**Source Lines**: app_backend.py (scattered in L5238-6350 -- /live-data, /tasks/<task_id>, /models, /rlhf/feedback, /debug/memory endpoints)
**Output File(s)**: `hg_backend/routes/data.py`
**Description**: Extract data-related routes into a Flask Blueprint named `data_bp`. Routes:
- `GET /live-data` -- returns live data from `LiveDataFetcher`
- `GET /tasks/<task_id>` -- returns task status from `TaskManager`
- `GET /models` -- returns available models from `Config.MODELS`
- `POST /rlhf/feedback` -- records RLHF feedback via `MemoryManager`
- `GET /debug/memory` -- returns memory debug info

Import from: `LiveDataFetcher`, `TaskManager`, `Config`, `MemoryManager`.
**Acceptance Criteria**:
- `from hg_backend.routes.data import data_bp` works
- All 5 routes are registered on the blueprint
- Each route returns proper JSON responses
**Dependencies**: Task 1, Task 2, Task 3, Task 4, Task 10, Task 15

---

### Task 30: Extract routes/memory.py
**Status**: [ ] Not Started
**Parallel Group**: E
**Source Lines**: app_backend.py (within L5238-6350 -- /memory GET/POST/DELETE, /memory/analyze GET/POST)
**Output File(s)**: `hg_backend/routes/memory.py`
**Description**: Extract memory management routes into a Flask Blueprint named `memory_bp`. Routes:
- `GET /memory` -- returns memory data
- `POST /memory` -- adds memory entries
- `DELETE /memory` -- clears specific memory sections
- `GET /memory/analyze` -- triggers memory analysis
- `POST /memory/analyze` -- triggers memory analysis with parameters

Import from: `MemoryManager`, `MemoryAnalysisWizard`.
**Acceptance Criteria**:
- `from hg_backend.routes.memory import memory_bp` works
- All memory CRUD routes work
- Analysis endpoint triggers `MemoryAnalysisWizard`
**Dependencies**: Task 1, Task 2, Task 3, Task 15, Task 25

---

### Task 31: Extract routes/chat.py
**Status**: [ ] Not Started
**Parallel Group**: E
**Source Lines**: app_backend.py (within L5238-6350 -- /chat, /debug/chat, /api/v1/memento-chat-working)
**Output File(s)**: `hg_backend/routes/chat.py`
**Description**: Extract agent chat routes into a Flask Blueprint named `chat_bp`. Routes:
- `POST /chat` -- Emissary chat endpoint (~140 lines, builds comprehensive context and calls Gemini)
- `POST /debug/chat` -- Dr. Debug chat endpoint (delegates to `debug_chat`)
- `POST /api/v1/memento-chat-working` -- Memento chat endpoint (delegates to `chat_with_memento`)

The `/chat` (Emissary) endpoint is the most complex -- it assembles memory context, live data, crawled data, agent activity, and tech stats, then calls `call_gemini_api`. Import from: `MemoryManager`, `SmartMemoryRetriever`, `LiveDataFetcher`, `VectorCache`, `call_gemini_api`, `Prompts`, `debug_chat`, `chat_with_memento`.
**Acceptance Criteria**:
- `from hg_backend.routes.chat import chat_bp` works
- All 3 chat routes are registered
- Emissary chat assembles full context and returns reply
- Debug chat delegates to `debug_chat` function
- Memento chat delegates to `chat_with_memento` function
**Dependencies**: Task 1, Task 2, Task 3, Task 6, Task 7, Task 10, Task 12, Task 13, Task 15, Task 19, Task 26

---

### Task 32: Extract routes/code_ops.py
**Status**: [ ] Not Started
**Parallel Group**: E
**Source Lines**: app_backend.py (within L5238-6350 -- /debug/analyze, /debug/rewrite)
**Output File(s)**: `hg_backend/routes/code_ops.py`
**Description**: Extract code operation routes into a Flask Blueprint named `code_ops_bp`. Routes:
- `POST /debug/analyze` -- delegates to `analyze_code_with_debugger`
- `POST /debug/rewrite` -- delegates to `rewrite_code_section`

Import from: `analyze_code_with_debugger`, `rewrite_code_section`, `MemoryManager`.
**Acceptance Criteria**:
- `from hg_backend.routes.code_ops import code_ops_bp` works
- Both routes registered and delegate correctly
- Proper error handling with JSON error responses
**Dependencies**: Task 1, Task 2, Task 3, Task 15, Task 19

---

### Task 33: Extract routes/projects.py
**Status**: [ ] Not Started
**Parallel Group**: E
**Source Lines**: app_backend.py (within L5238-6350 -- /generate-and-deploy ~120 lines, /evolve-app ~150 lines)
**Output File(s)**: `hg_backend/routes/projects.py`
**Description**: Extract project generation routes into a Flask Blueprint named `projects_bp`. Routes:
- `POST /generate-and-deploy` -- full project generation pipeline (idea generation OR user concept, code generation, evolution, deployment, memory storage)
- `POST /evolve-app` -- project evolution pipeline (evolution iterations, quality scoring, deployment)

These are the two most complex routes. They orchestrate the full pipeline: generate/receive concept -> generate code -> evolve code -> evaluate quality -> deploy to Netlify -> store in memory. Import from: `generate_frontend_code`, `evolve_app_code`, `evaluate_code_quality`, `generate_autonomous_idea`, `deploy_to_netlify_direct`, `robust_cleanup`, `add_watermark`, `MemoryManager`, `TaskManager`, `ClosedLoopLearning`, `Config`.
**Acceptance Criteria**:
- `from hg_backend.routes.projects import projects_bp` works
- Both routes registered with full pipeline logic
- Generation pipeline: concept -> code -> evolve -> evaluate -> deploy -> store
- Evolution pipeline: iterate -> evaluate -> deploy
- Task management for long-running operations
**Dependencies**: Task 1, Task 2, Task 3, Task 4, Task 15, Task 16, Task 17, Task 18, Task 20, Task 24

---

### Task 34: Extract routes/browser.py
**Status**: [ ] Not Started
**Parallel Group**: E
**Source Lines**: app_backend.py L5238-5310 (/browser/navigate), L5410-5840 (approximate -- /proxy ~350 lines, /proxy-form), (within L5238-6350 -- /browser/session, /browser/benni/chat)
**Output File(s)**: `hg_backend/routes/browser.py`
**Description**: Extract browser routes into a Flask Blueprint named `browser_bp`. Routes:
- `POST /browser/navigate` -- Playwright navigation with stealth scripts
- `GET /proxy` -- Ultra-fast proxy with intelligent routing (~350 lines, the largest single route). Handles URL fetching, Reddit detection, Playwright fallback, content-type routing, HTML rewriting
- `POST /proxy-form` -- Form submission proxy
- `POST /browser/session` -- Browser session creation
- `POST /browser/benni/chat` -- BENNI chat endpoint (delegates to `generate_benni_response`)

Import from: `BrowserIntelligence`, `RedditCommentExtractor`, `generate_benni_response`, `extract_page_content_advanced`, `fix_relative_urls`, `MemoryManager`, `Config`.
**Acceptance Criteria**:
- `from hg_backend.routes.browser import browser_bp` works
- All 5 browser routes registered
- Proxy handles all content types (HTML, images, CSS, JS)
- Playwright navigation works with stealth scripts
- Reddit comment extraction integrated in proxy
- BENNI chat delegates correctly
**Dependencies**: Task 1, Task 2, Task 3, Task 11, Task 15, Task 21, Task 22, Task 23

---

### Task 35: Banner and print_banner
**Status**: [ ] Not Started
**Parallel Group**: D
**Source Lines**: app_backend.py L50-64 (print_banner function)
**Output File(s)**: `hg_backend/logger.py` (append to existing)
**Description**: Add the `print_banner()` function to `logger.py`. This prints the ASCII art banner and version info. It should be a callable function, NOT auto-executed on import -- it will be called from `main.py` during startup.
**Acceptance Criteria**:
- `from hg_backend.logger import print_banner` works
- `print_banner()` prints the ASCII banner
- Banner is NOT printed on module import (only when explicitly called)
**Dependencies**: Task 3

---

### Task 36: Write routes/__init__.py (Blueprint Registration)
**Status**: [ ] Not Started
**Parallel Group**: Sequential
**Source Lines**: N/A (new file)
**Output File(s)**: `hg_backend/routes/__init__.py`
**Description**: Write the blueprint registration helper. Create a `register_blueprints(app)` function that imports all blueprints and registers them with the Flask app. This function is called from `app.py`.
```python
def register_blueprints(app):
    from .static import static_bp
    from .data import data_bp
    from .memory import memory_bp
    from .chat import chat_bp
    from .code_ops import code_ops_bp
    from .projects import projects_bp
    from .browser import browser_bp
    app.register_blueprint(static_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(memory_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(code_ops_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(browser_bp)
```
**Acceptance Criteria**:
- `from hg_backend.routes import register_blueprints` works
- `register_blueprints(app)` registers all 7 blueprints
- No import errors (all blueprint modules exist)
**Dependencies**: Tasks 28-34

---

### Task 37: Write app.py (Flask App Factory)
**Status**: [ ] Not Started
**Parallel Group**: Sequential
**Source Lines**: app_backend.py L69-70 (Flask app + CORS), L8965-8976 (if __name__ block)
**Output File(s)**: `hg_backend/app.py`
**Description**: Create the Flask app factory. Write a `create_app()` function that:
1. Creates Flask app instance
2. Configures CORS (same as original: `resources={r"/*": {"origins": "*"}}`)
3. Calls `register_blueprints(app)` from `routes/__init__`
4. Returns the app

Also write an `initialize()` function that:
1. Calls `print_banner()`
2. Initializes `VectorCache`
3. Initializes `BrowserIntelligence`
4. Initializes `ClosedLoopLearning`
5. Initializes `MemoryManager` and calls `load(force_reload=True)`
6. Starts `GrailCrawler` autonomous crawl (if desired)

The app factory pattern allows the same app to be used in testing and production.
**Acceptance Criteria**:
- `from hg_backend.app import create_app, initialize` works
- `app = create_app()` returns a Flask app with all routes registered
- `initialize()` performs all system initialization in correct order
- CORS is configured identically to original
**Dependencies**: Task 1, Task 2, Task 3, Task 6, Task 14, Task 15, Task 21, Task 24, Task 35, Task 36

---

### Task 38: Write main.py (Entry Point)
**Status**: [ ] Not Started
**Parallel Group**: Sequential
**Source Lines**: app_backend.py L8965-8976 (if __name__ == '__main__' block)
**Output File(s)**: `hg_backend/main.py`
**Description**: Create the entry point that replaces the monolith's `if __name__ == '__main__'` block. The file should:
1. Import `create_app` and `initialize` from `app.py`
2. Call `initialize()` to set up all subsystems
3. Call `create_app()` to get the Flask app
4. Run the app on `0.0.0.0:5000` with debug mode from env var

```python
#!/usr/bin/env python3
"""Holy Grail AI System - Modular Backend Entry Point"""

from hg_backend.app import create_app, initialize

if __name__ == '__main__':
    import os
    initialize()
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true"
    )
```
**Acceptance Criteria**:
- `python hg_backend/main.py` starts the Flask server on port 5000
- All routes are accessible (same URLs as original)
- Initialization runs in correct order (no import errors, no missing dependencies)
- Server accepts requests and responds correctly
**Dependencies**: Task 37

---

### Task 39: Write hg_backend/__init__.py (Package Exports)
**Status**: [ ] Not Started
**Parallel Group**: Sequential
**Source Lines**: N/A
**Output File(s)**: `hg_backend/__init__.py`
**Description**: Update the package `__init__.py` with version info and key exports. This makes the package importable and provides a clean public API.
```python
"""Holy Grail AI System - Modular Backend Package"""
__version__ = "4.0.0"
__author__ = "Dakota Rain Lock"

from .app import create_app
```
**Acceptance Criteria**:
- `import hg_backend` works
- `hg_backend.__version__` returns "4.0.0"
- `hg_backend.create_app` is accessible
**Dependencies**: Task 37

---

### Task 40: Smoke Test - Module Imports
**Status**: [ ] Not Started
**Parallel Group**: Sequential
**Source Lines**: N/A
**Output File(s)**: N/A (test only)
**Description**: Verify that every module in `hg_backend/` can be imported without error. Run:
```bash
python -c "
from hg_backend.config import Config, TurboConfig, ConsoleColors
from hg_backend.logger import print_error, print_success, print_info, print_warning, print_debug, print_banner, logger
from hg_backend.task_manager import TaskManager, request_pool, background_tasks
from hg_backend.token_pruner import TokenPruner
from hg_backend.vector_cache import VectorCache
from hg_backend.memory_retriever import SmartMemoryRetriever
from hg_backend.closed_loop_context import ClosedLoopLearningContext
from hg_backend.idea_validator import IdeaValidator
from hg_backend.live_data_fetcher import LiveDataFetcher
from hg_backend.content_extraction import extract_page_content_advanced, fix_relative_urls, handle_remove_readonly
from hg_backend.prompts import Prompts
from hg_backend.llm_client import call_gemini_api
from hg_backend.grail_crawler import GrailCrawler
from hg_backend.memory_manager import MemoryManager
from hg_backend.code_generation import generate_frontend_code, add_watermark, evaluate_code_quality
from hg_backend.idea_generation import generate_autonomous_idea
from hg_backend.code_evolution import evolve_app_code
from hg_backend.code_analysis import analyze_code_with_debugger, debug_chat, rewrite_code_section
from hg_backend.deployment import deploy_to_netlify_direct, robust_cleanup
from hg_backend.browser_intelligence import BrowserIntelligence
from hg_backend.reddit_extractor import RedditCommentExtractor
from hg_backend.benni_agent import generate_benni_response
from hg_backend.closed_loop_learning import ClosedLoopLearning
from hg_backend.memory_analysis import MemoryAnalysisWizard
from hg_backend.memory_agents import chat_with_memento, get_holy_grail_source_context
from hg_backend.agent_chunker import UniversalAgentChunker, create_universal_agent_wrapper
from hg_backend.routes import register_blueprints
from hg_backend.app import create_app, initialize
print('ALL IMPORTS SUCCESSFUL')
"
```
**Acceptance Criteria**:
- All imports succeed without error
- No circular import exceptions
- "ALL IMPORTS SUCCESSFUL" is printed
**Dependencies**: Tasks 1-39

---

### Task 41: Smoke Test - Flask App Starts
**Status**: [ ] Not Started
**Parallel Group**: Sequential
**Source Lines**: N/A
**Output File(s)**: N/A (test only)
**Description**: Verify that the Flask app starts and all routes are registered. Run:
```bash
python -c "
import os
os.environ.setdefault('GEMINI_API_KEY', 'test-key')
os.environ.setdefault('NETLIFY_AUTH_TOKEN', 'test-token')
from hg_backend.app import create_app
app = create_app()
rules = [rule.rule for rule in app.url_map.iter_rules()]
print(f'Routes registered: {len(rules)}')
required_routes = ['/chat', '/proxy', '/memory', '/debug/analyze', '/debug/chat', '/debug/rewrite', '/generate-and-deploy', '/evolve-app', '/live-data', '/models', '/browser/navigate', '/browser/benni/chat', '/api/v1/memento-chat-working', '/rlhf/feedback']
for route in required_routes:
    assert route in rules, f'Missing route: {route}'
print('ALL ROUTES VERIFIED')
"
```
**Acceptance Criteria**:
- Flask app creates without error
- All 14+ required routes are registered
- No missing routes
**Dependencies**: Task 40

---

### Task 42: Smoke Test - Endpoint Responses
**Status**: [ ] Not Started
**Parallel Group**: Sequential
**Source Lines**: N/A
**Output File(s)**: N/A (test only)
**Description**: Verify that key endpoints return valid responses using Flask's test client. Test:
1. `GET /` -- returns 200 or redirect
2. `GET /models` -- returns JSON with model list
3. `GET /live-data` -- returns JSON (may have empty data without API keys)
4. `POST /chat` with empty conversation -- returns JSON response
5. `POST /debug/analyze` with sample code -- returns JSON response
6. `GET /memory` -- returns JSON memory data

Use the Flask test client (`app.test_client()`), not actual HTTP requests.
**Acceptance Criteria**:
- All tested endpoints return HTTP 200 or expected status codes
- JSON responses have correct structure
- No 500 errors on basic requests
**Dependencies**: Task 41

---

### Task 43: Final Validation - Behavioral Comparison
**Status**: [ ] Not Started
**Parallel Group**: Sequential
**Source Lines**: N/A
**Output File(s)**: N/A (test only)
**Description**: Compare behavior of old monolith vs new modular app. For each of these operations, verify the new app produces equivalent output:
1. Route listing (same URLs available)
2. Config values (same API keys, paths, model names)
3. Memory file operations (same JSON structure)
4. Prompt templates (same strings)

This is a manual verification task. Start both apps (on different ports) and compare responses, or use the Flask test client on both.
**Acceptance Criteria**:
- Route sets are identical between old and new
- Config values match
- Memory JSON structure matches
- Prompt templates are byte-for-byte identical
- All function signatures match (same parameters, same defaults)
**Dependencies**: Task 42

---

## Parallel Execution Groups

Tasks are organized into parallel groups. All tasks within a group can run simultaneously in separate worktrees or terminals.

| Group | Tasks | Description | Prerequisite |
|-------|-------|-------------|--------------|
| **Sequential** | 1 | Directory structure | None |
| **A** | 2, 3, 4, 5 | Config, Logger, TaskManager, TokenPruner (leaf modules, no cross-deps) | Task 1 |
| **B** | 6, 7, 8, 9, 10, 11 | VectorCache, MemoryRetriever, ClosedLoopContext, IdeaValidator, LiveData, ContentExtraction (utility layer) | Tasks 1-3 |
| **C** | 12, 13, 14 | Prompts, LLM Client, GrailCrawler (depend on A+B) | Tasks 1-6 |
| **Sequential** | 15 | MemoryManager (depends on most of B+C, many things depend on it) | Tasks 1-7, 13 |
| **D** | 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 35 | Code gen, agents, browser, deployment (depend on MemoryManager) | Task 15 |
| **E** | 28, 29, 30, 31, 32, 33, 34 | All route blueprints (depend on D layer functions) | Tasks in D they reference |
| **Sequential** | 36, 37, 38, 39 | Blueprint registration, app factory, main.py, package init | All E tasks |
| **Sequential** | 40, 41, 42, 43 | Smoke tests and validation | Task 39 |

---

## Optimized Execution Timeline

```
Task 1 (dirs)
    |
    +---> Tasks 2,3,4,5 (parallel group A)
    |         |
    |         +---> Tasks 6,7,8,9,10,11 (parallel group B)
    |                   |
    |                   +---> Tasks 12,13,14 (parallel group C)
    |                             |
    |                             +---> Task 15 (MemoryManager - sequential)
    |                                       |
    |                                       +---> Tasks 16-27,35 (parallel group D)
    |                                                 |
    |                                                 +---> Tasks 28-34 (parallel group E)
    |                                                           |
    |                                                           +---> Tasks 36,37,38,39 (sequential)
    |                                                                     |
    |                                                                     +---> Tasks 40,41,42,43 (sequential)
```

---

## Dependency Graph (Condensed)

```
config.py (2) ─────────────────────────┐
logger.py (3) ─────────────────────────┤
task_manager.py (4) ───────────────────┤
token_pruner.py (5) ───────────────────┤
                                       ├──> memory_manager.py (15)
vector_cache.py (6) ──────────────────┤         │
memory_retriever.py (7) ──────────────┤         │
closed_loop_context.py (8) ───────────┤         ├──> code_generation.py (16)
idea_validator.py (9) ────────────────┤         ├──> idea_generation.py (17)
live_data_fetcher.py (10) ────────────┤         ├──> code_evolution.py (18)
content_extraction.py (11) ───────────┤         ├──> code_analysis.py (19)
prompts.py (12) ──────────────────────┤         ├──> benni_agent.py (23)
llm_client.py (13) ───────────────────┤         ├──> memory_agents.py (26)
grail_crawler.py (14) ────────────────┘         │
                                                ├──> routes/*.py (28-34)
deployment.py (20) ────────────────────────────┤
browser_intelligence.py (21) ──────────────────┤
reddit_extractor.py (22) ─────────────────────┤
closed_loop_learning.py (24) ─────────────────┤
memory_analysis.py (25) ──────────────────────┤
agent_chunker.py (27) ────────────────────────┘
                                                │
                                                v
                                routes/__init__.py (36)
                                       │
                                       v
                                    app.py (37)
                                       │
                                       v
                                   main.py (38)
```

---

## Summary

| Metric | Value |
|--------|-------|
| Total Tasks | 43 |
| Module Extraction Tasks | 27 (Tasks 2-27, 35) |
| Route Blueprint Tasks | 7 (Tasks 28-34) |
| Infrastructure Tasks | 5 (Tasks 1, 36-39) |
| Test/Validation Tasks | 4 (Tasks 40-43) |
| Max Parallel Width | 13 (Group D) |
| Estimated Total Effort | ~40-50 focused-code-writer agent sessions |
| Dead Code Eliminated | ~2,500 lines (monkey patches, duplicates, superseded versions) |
| Expected New Package Size | ~6,500 lines (28% reduction from 9,008) |
