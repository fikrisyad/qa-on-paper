# qa-on-paper
QA system based on a PDF input paper.

## Usage
### Install the dependencies
`pip install -r requirements.txt`

### Set up the environment variables
Create a .env file in the root directory of the project and add the following line, replacing <OPENAI_API_KEY> with your OpenAI API key:\
`OPENAI_API_KEY=<OPENAI_API_KEY>
`

### Run via CLI 
This QA system accept pdf input via URL and a string of question (make sure to enclose the question inside quotes).
```commandline
>> python main.py <pdf url> <question>
<answer>
```

Example:
```commandline
>> python main.py https://arxiv.org/pdf/2304.03442.pdf "What architecture was proposed by ths paper?"
The paper proposed a novel architecture that enables generative agents to remember, retrieve, reflect, interact with other agents, and plan through dynamically evolving circumstances. The architecture leverages the powerful prompting capabilities of large language models and supplements those capabilities to support longer-term agent coherence, the ability to manage dynamically-evolving memory, and recursively produce more generations.
```

#### Note
This QA system uses OpenAI Embeddings API as the default embedder.\
You can change the embedder with open source ones from Hugging Face repo, e.g. `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
