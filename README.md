# Topic Classification

## Install

1. Clone Repository
   
   ```
   git clone https://github.com/ShihHsuanChen/emotiondet
   cd emotiondet 
   ```

2. Install `uv`

   ```
   pip install uv
   ```

   See https://docs.astral.sh/uv/getting-started/installation/#standalone-installer for more options

3. Install Dependencies

   ```
   uv sync
   ```

4. Acitvate the Environment
   
   ```
   source .venv/bin/activate
   ```

5. Configuration

   1. Copy `.env.example` to `.env`
   2. Setup `GEMINI_API_KEY` obtained from [Google AI Studio](https://aistudio.google.com/apikey)
   

## Run Application

### Simply Run

```
classify-topics <input json file>
```

or 

```
python main.py <input json file>
```

or 

```
uv run main.py <input json file>
```

### More Options

```
usage: classify-topics [-h] [-o OUTPUT] [--retry RETRY] [--retry-delay RETRY_DELAY] [-q] [--debug] file

Classify topics of the given sentences according to the sentences and make a summary.

positional arguments:
  file                  input json file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output json path (default: None)
  --retry RETRY         maximum retry counts (default: 3)
  --retry-delay RETRY_DELAY
                        time interval in second between each call to llm (default: 1)
  -q, --quiet           don't print any processing message (default: False)
  --debug               debug mode (default: False)
```

### Troubleshooting

Now this app has high probability to get the `MaximumRetryError` due to the Gemini's API response.
If happened, please try to increase the value of `--retry <retry>` and `--retry-delay <retry-delay>`.


## TODO

- JSON generation from Gemini API (free) seems not quite stable, try to use other LLM API.
- Try to classify the sentences using embedding model and clustering algorithm (such as T-SEN) instead of LLM.
