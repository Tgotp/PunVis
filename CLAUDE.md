# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

成语双关语可视化Agent (PunVis) - An agent system that transforms Chinese idioms (成语) into visual puns using homophonic substitutions. The system features reflective learning: it saves success/failure experiences after each attempt and uses historical learnings to improve future generations.

## Running the Project

### Initial Setup
```bash
# 1. Create config from template
cp src/config_template.py src/config.py
# Edit src/config.py and add your OpenAI API key

# 2. Install dependencies
pip install -r requirements.txt
```

### Running the Agent
```bash
# Using the run script
./run.sh --idiom "机不可失"

# Or directly
python src/main.py --idiom "机不可失"

# With custom parameters
python src/main.py --idiom "机不可失" --max-iterations 5 --min-confidence 0.8
```

## Architecture

### Core Components

**1. Agent Loop** (`src/agent.py` - `PunVisAgent.generate_with_reflection()`)
The main workflow follows this pattern:
- Load historical experiences from memory
- Generate pun + scene description using LLM (GPT-4o)
- Validate pun format (at least 1 character substituted, same length)
- Generate image via `z-image-turbo` model
- VLM (GPT-4o vision) evaluates the image - guesses the original idiom
- Reflect on success/failure and extract learnings
- Save experience to memory if improved over previous attempt

**2. Memory System** (`src/memory.py` - `ExperienceMemory`)
- Stores experiences in `memory/experiences.json`
- Stores extracted rules in `memory/rules.json`
- Tracks: idiom, pun, scene, success/failure, iteration count, key factors
- Provides success pattern analysis for prompt context

**3. Validation & Reflection** (`src/agent.py`)
- `check_pun_valid()`: Validates pun has at least 1 character different but same length
- `_reflect()`: Analyzes why VLM guessed correctly/incorrectly
- `_save_experience()`: Only saves if improvement over previous attempt
- `_extract_rules_from_reflection()`: Uses LLM to extract reusable rules

### File Structure

```
src/
  main.py       # Entry point: argument parsing, client init, runs agent
  agent.py      # Core PunVisAgent class with reflection loop
  memory.py     # ExperienceMemory for persistent learning
  config.py     # API keys (user-created from config_template.py)

memory/
  experiences.json   # Saved experience records
  rules.json         # Extracted generation rules

examples/
  chengyu_examples.md   # Reference examples loaded into prompts
  PunBenchmark.json     # Test dataset

output/images/   # Generated images saved as {idiom}_v{iteration}.png
```

### Key Constraints

**Pun Requirements** (enforced in `check_pun_valid()`):
- Must replace at least 1 character with homophone
- Must maintain same character count as original idiom
- Homophone check is done by LLM, not code

**Scene Requirements** (enforced in prompts):
- 80-120 Chinese characters
- NO text/letters/signs in scene (visual elements only)
- Must express pun meaning through visual action/scene

**Image Generation**:
- Uses `z-image-turbo` model via OpenAI-compatible API
- Size: 1024x1024
- Style: "Bright cartoon style, clean background, simple composition"

### Configuration (`src/config.py`)

```python
OPENAI_API_KEY = "your-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"  # Or proxy like jusuan.ai
TEXT_MODEL = "gpt-4o"
VISION_MODEL = "gpt-4o"
MAX_ITERATIONS = 5
MIN_CONFIDENCE = 0.8
```

### Experience Format

Experiences saved to `memory/experiences.json`:
```json
{
  "idiom": "机不可失",
  "pun": "鸡不可湿",
  "scene_zh": "一只小鸡在下雨天撑着伞...",
  "success": true,
  "iteration": 2,
  "reason": "主体明确（小鸡撑伞），动作清晰（避雨）",
  "key_factors": ["主体明确", "动作清晰"],
  "timestamp": "2024-01-15T10:30:00",
  "vlm_feedback": "看到了小鸡和伞，联想到鸡..."
}
```

## Common Development Tasks

### Viewing Learned Rules
```bash
cat memory/rules.json
```

### Viewing Experience Statistics
Experiences are printed after each run. Programmatically:
```python
from memory import ExperienceMemory
mem = ExperienceMemory(".")
print(mem.get_statistics())
```

### Adding Reference Examples
Edit `examples/chengyu_examples.md` - the first 5 examples are loaded into prompts.

### Testing Without Image Generation
The agent has `skip_image` parameter (not exposed in main.py) for testing pun generation without DALL-E calls.
