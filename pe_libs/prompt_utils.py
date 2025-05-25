# Copyright 2025 SLAPaper
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

# extended from modules/prompt_parser.py
# handle step schedule format like [prompt:step1:step2] or [prompt1|prompt2|prompt3]
re_attention = re.compile(
    r"""
# 1. 新增：匹配形如 [text1:text2:step] 或 [prompt:step1:step2] 或 [prompt1|prompt2|prompt3] 的块, 捕获方括号内的内容
(?P<schedule>\[[^\]]+:[^:\]]*:[^\]]*\])|
(?P<alternate>\[[^\|\]]+(\|[^\|\]]*)+\])|
# 2. 转义字符 (优先级高于普通括号)
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
# 3. 普通的左圆括号或左方括号 (用于权重调整)
\(|
\[|
# 4. 带权重的右圆括号，例如 (text:1.23) 中的 :1.23)
:\s*(?P<weight>[+-]?[.\d]+)\s*\)|
\)|
]|
# 6. 不包含特殊字符的普通文本
[^\\()\[\]:]+|
# 7. 单独的冒号 (如果未被前面的模式捕获)
:
""",
    re.X,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text):
    r"""
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
    - (abc) - increases attention to abc by a multiplier of 1.1
    - (abc:3.12) - increases attention to abc by a multiplier of 3.12
    - [abc] - decreases attention to abc by a multiplier of 1.1
    - [prompt:step1:step2] - as plain text "[prompt:step1:step2]"
    - [prompt1|prompt2] - as plain text "[prompt1|prompt2]"
    - \( - literal character '('
    - \[ - literal character '['
    - \) - literal character ')'
    - \] - literal character ']'
    - \\ - literal character '\'
    - anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        token_str = m.group(0)  # 完整的匹配项

        # 检查捕获组
        literal_square_block_content = m.group('schedule')
        literal_square_block_content2 = m.group('alternate')
        explicit_weight_str = m.group('weight')

        if literal_square_block_content is not None:
            # 匹配到了 [text:with:colons] 格式
            res.append([literal_square_block_content, 1.0])
        elif literal_square_block_content2 is not None:
            # 匹配到了 [prompt1|prompt2] 格式
            res.append([literal_square_block_content2, 1.0])
        elif token_str.startswith("\\"):
            # 处理转义字符
            res.append([token_str[1:], 1.0])
        elif token_str == "(":
            round_brackets.append(len(res))
        elif token_str == "[":
            # 这是普通的 '[' (未被 literal_square_block_content 捕获)
            square_brackets.append(len(res))
        elif explicit_weight_str is not None and round_brackets:
            # 匹配到了 (text:weight) 中的 :weight) 部分
            # token_str 此时是类似 ":1.23)"
            # explicit_weight_str 是 "1.23"
            if not round_brackets:  # 安全检查，理论上不应发生
                # Malformed: Orphaned :weight) without preceding (
                # Add as literal text, or raise error. For now, add as literal.
                parts = re.split(re_break, token_str)
                for i, part in enumerate(parts):
                    if i > 0:
                        res.append(["BREAK", -1])
                    if part:
                        res.append([part, 1.0])
                continue  # Skip multiply_range

            multiply_range(round_brackets.pop(), float(explicit_weight_str))
        elif token_str == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif token_str == "]" and square_brackets:
            # 这是普通的 ']' (未被 literal_square_block_content 捕获)
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            # 普通文本或未处理的冒号
            # token_str 是匹配到的普通文本片段，例如 "abc" 或者单独的 ":"
            parts = re.split(re_break, token_str)
            for i, part in enumerate(parts):
                if i > 0:  # A BREAK was found by re.split
                    res.append(["BREAK", -1])
                if part:  # 只添加非空文本片段
                    res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res
