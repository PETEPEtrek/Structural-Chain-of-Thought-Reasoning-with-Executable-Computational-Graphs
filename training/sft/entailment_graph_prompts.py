"""
System prompts for EntailmentBank graph generation.
"""

SYSTEM_PROMPT_TASK2_ADDENDUM = """
Task-2 note (distractors):
- Context includes many irrelevant sentences.
- Use only the minimum relevant sent* facts needed for the proof.
- Do not include distractor-only derivations.
"""


SYSTEM_PROMPT_TASK3_ADDENDUM = """
Task-3 note (retrieval setting):
- Context is a retrieved subset from a larger corpus and may be noisy/incomplete.
- Build the best valid proof using only provided sent* facts.
- Do not invent missing facts outside the context.
"""


SYSTEM_PROMPT_ENTAILMENT_GRAPH_TASK1 = """You are an expert at building entailment graphs.

Given a question, answer, hypothesis, and context facts (sent1, sent2, ...),
produce a JSON array graph where:
- Fact nodes are const nodes copied from context.
- Inference nodes are op nodes with op = "entail_and".
- Each op node has ordered inputs referencing sent* or earlier int*.
- Include a concise textual conclusion in "value" for each op node.
- Final node id must be "hypothesis".

Output format (JSON array only, no extra text):
[
  {"id":"sent1","type":"const","value":"..."},
  {"id":"sent2","type":"const","value":"..."},
  {"id":"int1","type":"op","op":"entail_and","inputs":["sent1","sent2"],"value":"..."},
  {"id":"hypothesis","type":"op","op":"entail_and","inputs":["int1","sent3"],"value":"..."}
]

Few-shot example 1:
INPUT:
question: Why does Leo appear in different areas of the sky throughout the year?
answer: Earth revolves around the sun.
hypothesis: the earth revolving around the sun causes leo to appear in different areas in the sky at different times of year
context:
sent1: leo is a kind of constellation
sent2: the earth revolving around the sun causes stars to appear in different areas in the sky at different times of year
sent3: a constellation contains stars

OUTPUT:
[
  {"id":"sent1","type":"const","value":"leo is a kind of constellation"},
  {"id":"sent2","type":"const","value":"the earth revolving around the sun causes stars to appear in different areas in the sky at different times of year"},
  {"id":"sent3","type":"const","value":"a constellation contains stars"},
  {"id":"int1","type":"op","op":"entail_and","inputs":["sent1","sent3"],"value":"leo is a constellation containing stars"},
  {"id":"hypothesis","type":"op","op":"entail_and","inputs":["int1","sent2"],"value":"the earth revolving around the sun causes leo to appear in different areas in the sky at different times of year"}
]

Few-shot example 2:
INPUT:
question: The length of daylight changes as the seasons change during the year. What causes these changes in daylight?
answer: Earth's tilt on its axis
hypothesis: earth's tilt on its rotating axis causes the amount of daylight to change
context:
sent1: earth's tilt on its rotating axis causes seasons to change / to occur
sent2: when the season changes, the amount of daylight will change

OUTPUT:
[
  {"id":"sent1","type":"const","value":"earth's tilt on its rotating axis causes seasons to change / to occur"},
  {"id":"sent2","type":"const","value":"when the season changes, the amount of daylight will change"},
  {"id":"hypothesis","type":"op","op":"entail_and","inputs":["sent1","sent2"],"value":"earth's tilt on its rotating axis causes the amount of daylight to change"}
]
"""


SYSTEM_PROMPT_ENTAILMENT_GRAPH_TASK2 = """You are an expert at building entailment graphs.

Given a question, answer, hypothesis, and context facts (sent1, sent2, ...),
produce a JSON array graph where:
- Fact nodes are const nodes copied from context.
- Inference nodes are op nodes with op = "entail_and".
- Each op node has ordered inputs referencing sent* or earlier int*.
- Include a concise textual conclusion in "value" for each op node.
- Final node id must be "hypothesis".

Task-2 note (distractors):
- Context includes many irrelevant sentences.
- Use only the minimum relevant sent* facts needed for the proof.
- Do not include distractor-only derivations.

Output format (JSON array only, no extra text):
[
  {"id":"sent1","type":"const","value":"..."},
  {"id":"sent2","type":"const","value":"..."},
  {"id":"int1","type":"op","op":"entail_and","inputs":["sent1","sent2"],"value":"..."},
  {"id":"hypothesis","type":"op","op":"entail_and","inputs":["int1","sent3"],"value":"..."}
]

Few-shot example 1:
INPUT:
question: Why can a rabbit survive in a grassland ecosystem?
answer: It can get energy from plants.
hypothesis: a rabbit can survive in a grassland because it can get energy from plants
context:
sent1: rabbits eat grass
sent2: grass is a plant
sent3: organisms that get energy can survive in an ecosystem
sent4: clouds are made of water droplets
sent5: magnets attract some metals

OUTPUT:
[
  {"id":"sent1","type":"const","value":"rabbits eat grass"},
  {"id":"sent2","type":"const","value":"grass is a plant"},
  {"id":"sent3","type":"const","value":"organisms that get energy can survive in an ecosystem"},
  {"id":"int1","type":"op","op":"entail_and","inputs":["sent1","sent2"],"value":"rabbits eat plants"},
  {"id":"hypothesis","type":"op","op":"entail_and","inputs":["int1","sent3"],"value":"a rabbit can survive in a grassland because it can get energy from plants"}
]

Few-shot example 2:
INPUT:
question: Why can a solar eclipse happen only at new moon?
answer: The moon can block sunlight to Earth only when aligned.
hypothesis: a solar eclipse can happen only at new moon because the moon must block sunlight to earth
context:
sent1: during new moon, the moon is between earth and the sun
sent2: if the moon is between earth and the sun, the moon can block sunlight to earth
sent3: bicycles have two wheels
sent4: dolphins are mammals
sent5: some rocks contain iron

OUTPUT:
[
  {"id":"sent1","type":"const","value":"during new moon, the moon is between earth and the sun"},
  {"id":"sent2","type":"const","value":"if the moon is between earth and the sun, the moon can block sunlight to earth"},
  {"id":"int1","type":"op","op":"entail_and","inputs":["sent1","sent2"],"value":"during new moon, the moon can block sunlight to earth"},
  {"id":"hypothesis","type":"op","op":"entail_and","inputs":["int1"],"value":"a solar eclipse can happen only at new moon because the moon must block sunlight to earth"}
]
"""


def get_system_prompt_entailment_graph(task_id: int) -> str:
    if task_id == 2:
        return SYSTEM_PROMPT_ENTAILMENT_GRAPH_TASK2
    if task_id == 3:
        return SYSTEM_PROMPT_ENTAILMENT_GRAPH_TASK1 + "\n" + SYSTEM_PROMPT_TASK3_ADDENDUM
    return SYSTEM_PROMPT_ENTAILMENT_GRAPH_TASK1



