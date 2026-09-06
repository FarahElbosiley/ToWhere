# ToWhere - AI Trip Planner

A LangChain-based tourism agent that plans day-by-day itineraries by
combining live place/weather data with a small local knowledge base of
practical travel tips (RAG).

## Architecture

```
User conversation (questions, requests, revisions)
        │
        ▼
Streamlit chat UI → ConversationService
  │
  ▼
LangChain tool-calling agent with SQLite chat history
        │
        ├── search_places          → OpenTripMap API (attractions, food, museums)
        ├── get_weather_forecast   → OpenWeatherMap API
        └── search_travel_guide    → Chroma vector store over data/city_guides/*.md
        │
        ▼
      Free-text planning notes or conversational reply
        │
        ▼
      Structured-output LLM call when a plan is created or revised
        │
        ▼
Streamlit UI renders the itinerary as markdown
```

The agent runs in two stages: a tool-calling loop that gathers real
information and reasons in free text, then a second call that converts
marked planning responses into a strictly-typed `Itinerary` object. The
conversation service wraps the agent with LangChain's
`RunnableWithMessageHistory` and persists each session in local SQLite, so
follow-up requests such as "make day 2 lighter" can revise the existing
plan. Clarifying questions remain ordinary chat replies.

## Project layout

```
config.py           Centralized settings / env loading
agent.py            TourismAgent: tool-calling agent + structuring step
conversation_service.py  Persistent chat orchestration and turn results
itinerary.py        Itinerary pydantic model + markdown renderer
app.py              Streamlit presentation layer
tools/
  places.py         OpenTripMap attraction/restaurant search
  weather.py        OpenWeatherMap forecast
  rag.py            Chroma-backed retriever over city guide docs
data/city_guides/     Source markdown files for the RAG corpus
conversations.sqlite   Local persistent chat history (gitignored)
```

## Cost: $0

Every service this project uses has a genuinely free tier — no credit
card required anywhere, and everything runs in the cloud (no local model
install needed):

| Service | Role | Free tier |
|---|---|---|
| [OpenRouter](https://openrouter.ai/keys) | LLM (MiniMax M3) | `:free` model variant, $0/M tokens, tool-calling supported |
| [OpenTripMap](https://opentripmap.io/product) | Places/attractions search | Free API key, no billing setup |
| [OpenWeatherMap](https://openweathermap.org/api) | Weather forecast | Free tier, no billing setup |
| Chroma + HuggingFace embeddings | RAG vector store | Fully local computation, no API calls at all |

The one thing to know: OpenRouter's free models are rate-limited (fine for
building and demoing an agent, not for high-volume production traffic).
Check current limits at [openrouter.ai/docs](https://openrouter.ai/docs).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
```

Required keys:
- `OPENROUTER_API_KEY` — required, powers the agent (free, see table above)
- `OPENTRIPMAP_API_KEY` — free tier, powers place search
- `OPENWEATHER_API_KEY` — free tier, powers weather forecasts

The app still runs without the two data-source keys, but those tools will
return an error message instead of real data — the RAG tool works
regardless since it's local.

## Run

```bash
python3 -m streamlit run app.py
```

The app uses a stable UUID stored in the Streamlit browser session as the
history key. Chat messages are persisted in `conversations.sqlite`; the
database is local and gitignored, like the Chroma database.

Use the sidebar to switch between saved conversations or create a new chat.
Each conversation has its own persisted LangChain message history and is
listed by a short title derived from its first user message.

## Extending this project

- **Add more cities**: drop another markdown file into `data/city_guides/`
  and delete `data/chroma_db/` so it re-indexes on next run.
- **Add booking tools**: a flight/hotel search tool would slot in next to
  `search_places` with no changes to the agent loop.
- **Swap the LLM**: `agent.py` is the only file that constructs the chat
  model — swap `ChatOpenAI`/`base_url` for any other LangChain chat model,
  including a fully local one (e.g. Ollama) if you ever want zero API
  calls at all.
