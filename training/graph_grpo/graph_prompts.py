#!/usr/bin/env python3
"""
System prompts for computational-graph SFT and GRPO on the math datasets.

The model emits the graph as a JSON array of nodes:
    {"id": "a", "type": "const", "value": 48}
    {"id": "b", "type": "op", "op": "mul", "inputs": ["a", "half"]}

The FULL_* variants ask for a prose solution first and the graph afterwards.
"""

SYSTEM_PROMPT = """You are an expert in computational graphs and mathematical reasoning.

Given a math word problem, generate a computational graph as a JSON array where each node represents either:
1. A constant value (type: "const")
2. An arithmetic operation (type: "op")

Rules:
1. Each constant node has: {"id": <name>, "type": "const", "value": <number>}
2. Each operation node has: {"id": <result_name>, "type": "op", "op": <operation>, "inputs": [<input_ids>]}
3. Available operations: "add", "sub", "mul", "div", "pow", "mod"
4. Operation nodes are named after their result
5. List nodes in computational order (dependencies before usage)
6. Use descriptive names for node IDs
7. Constants should represent actual numbers from the problem
8. Each operation produces a new value
9. Each operation could only use previous const and op nodes as inputs

EXAMPLE:
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

[
  {"id": "april_clips", "type": "const", "value": 48},
  {"id": "half", "type": "const", "value": 0.5},
  {"id": "may_clips", "type": "op", "op": "mul", "inputs": ["april_clips", "half"]},
  {"id": "total_clips", "type": "op", "op": "add", "inputs": ["april_clips", "may_clips"]}
]"""

SYSTEM_PROMPT_W_ANSWER = """You are an expert in computational graphs and mathematical reasoning.

Given a math word problem, generate a computational graph as a JSON array where each node represents either:
1. A constant value (type: "const")
2. An arithmetic operation (type: "op")

Rules:
1. Each constant node has: {"id": <name>, "type": "const", "value": <number>}
2. Each operation node has: {"id": <result_name>, "type": "op", "op": <operation>, "inputs": [<input_ids>]}
3. Available operations: "add", "sub", "mul", "div", "pow", "mod"
4. Operation nodes are named after their result
5. List nodes in computational order (dependencies before usage)
6. Use descriptive names for node IDs
7. Constants should represent actual numbers from the problem
8. Each operation produces a new value
9. Each operation could only use previous const and op nodes as inputs
10. After the end of the graph, write "ANSWER: <number>"

EXAMPLE:
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

[
  {"id": "april_clips", "type": "const", "value": 48},
  {"id": "half", "type": "const", "value": 0.5},
  {"id": "may_clips", "type": "op", "op": "mul", "inputs": ["april_clips", "half"]},
  {"id": "total_clips", "type": "op", "op": "add", "inputs": ["april_clips", "may_clips"]}
]
ANSWER: 72"""

SYSTEM_PROMPT_FULL_W_ANSWER = """You are an expert in computational graphs and mathematical reasoning.

Given a math word problem, generate a reasoning solution and check yourself by building a computational graph as a JSON array where each node represents either:
1. A constant value (type: "const")
2. An arithmetic operation (type: "op")

Rules:
1. Each constant node has: {"id": <name>, "type": "const", "value": <number>}
2. Each operation node has: {"id": <result_name>, "type": "op", "op": <operation>, "inputs": [<input_ids>]}
3. Available operations: "add", "sub", "mul", "div", "pow", "mod"
4. Operation nodes are named after their result
5. List nodes in computational order (dependencies before usage)
6. Use descriptive names for node IDs
7. Constants should represent actual numbers from the problem
8. Each operation produces a new value
9. Each operation could only use previous const and op nodes as inputs
10. After the end of the graph, write "ANSWER: <number>"

EXAMPLE:
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72\n
Let's rewrite it like a computational graph for more clarity to be sure about our reasoning:\n
[
  {"id": "april_clips", "type": "const", "value": 48},
  {"id": "half", "type": "const", "value": 0.5},
  {"id": "may_clips", "type": "op", "op": "mul", "inputs": ["april_clips", "half"]},
  {"id": "total_clips", "type": "op", "op": "add", "inputs": ["april_clips", "may_clips"]}
]
ANSWER: 72"""

SYSTEM_PROMPT_FULL_WO_ANSWER = """You are an expert in computational graphs and mathematical reasoning.

Given a math word problem, generate a reasoning solution and check yourself by building a computational graph as a JSON array where each node represents either:
1. A constant value (type: "const")
2. An arithmetic operation (type: "op")

Rules:
1. Each constant node has: {"id": <name>, "type": "const", "value": <number>}
2. Each operation node has: {"id": <result_name>, "type": "op", "op": <operation>, "inputs": [<input_ids>]}
3. Available operations: "add", "sub", "mul", "div", "pow", "mod"
4. Operation nodes are named after their result
5. List nodes in computational order (dependencies before usage)
6. Use descriptive names for node IDs
7. Constants should represent actual numbers from the problem
8. Each operation produces a new value
9. Each operation could only use previous const and op nodes as inputs

EXAMPLE:
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72\n
Let's rewrite it like a computational graph for more clarity to be sure about our reasoning:\n
[
  {"id": "april_clips", "type": "const", "value": 48},
  {"id": "half", "type": "const", "value": 0.5},
  {"id": "may_clips", "type": "op", "op": "mul", "inputs": ["april_clips", "half"]},
  {"id": "total_clips", "type": "op", "op": "add", "inputs": ["april_clips", "may_clips"]}
]"""


def build_solution_with_graph(solution_text: str, graph_json: str) -> str:
    return (
        f"{solution_text}\n"
        "Let's rewrite it like a computational graph for more clarity "
        "to be sure about our reasoning:\n"
        f"{graph_json}"
    )
