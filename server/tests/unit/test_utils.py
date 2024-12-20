from apis.base_client import State
from apis.utils import parse_response


def test_parse_response_basic():
    completion = "Start\nLine 1\nLine 2\nLine 3\nStop"
    max_lines = 10
    stop_tokens = ["Stop"]
    start_phrases = ["Start"]
    state = State(prefix="", suffix="")

    result = parse_response(completion, max_lines, stop_tokens, start_phrases, state)
    assert result == "\nLine 1\nLine 2\nLine 3\n"


def test_parse_response_no_start_phrase():
    completion = "Line 1\nLine 2\nLine 3\nStop"
    max_lines = 10
    stop_tokens = ["Stop"]
    start_phrases = ["NonexistentStart"]
    state = State(prefix="", suffix="")

    result = parse_response(completion, max_lines, stop_tokens, start_phrases, state)
    assert result == "Line 1\nLine 2\nLine 3\n"


def test_parse_response_no_stop_token():
    completion = "Start\nLine 1\nLine 2\nLine 3\nLine 4"
    max_lines = 10
    stop_tokens = ["NonexistentStop"]
    start_phrases = ["Start"]
    state = State(prefix="", suffix="")

    result = parse_response(completion, max_lines, stop_tokens, start_phrases, state)
    assert result == "\nLine 1\nLine 2\nLine 3\nLine 4"


def test_parse_response_with_overlap():
    completion = "Prefix\nMiddle\nSuffix"
    max_lines = 10
    stop_tokens = []
    start_phrases = []
    state = State(prefix="Prefix", suffix="Suffix")
    prompt_options = ["overlap"]

    result = parse_response(
        completion, max_lines, stop_tokens, start_phrases, state, prompt_options
    )
    assert result == "\nMiddle\n"


def test_parse_response_with_overlap_newline():
    completion = "_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n    return result\n"
    max_lines = 10
    stop_tokens = []
    start_phrases = []
    state = State(
        prefix="from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n    result = []\n    current_string = []\n    current_depth = 0\n\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n\n            if current",
        suffix="esult",
    )
    prompt_options = ["overlap"]

    result = parse_response(
        completion, max_lines, stop_tokens, start_phrases, state, prompt_options
    )
    assert (
        result
        == "_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n    return r"
    )


def test_parse_response_with_overlap_newline_2():
    completion = 'f"JSONDecodeError: {l}")\n                    continue\n                if row["type"] in VOTES:'
    max_lines = 10
    stop_tokens = []
    start_phrases = []
    state = State(
        prefix="                    print(",
        suffix="""                    continue
                if row["type"] in VOTES:
                    data.append(row)
            break
        except FileNotFoundError:
            time.sleep(2)
    return data

""",
    )
    prompt_options = ["overlap"]

    result = parse_response(
        completion, max_lines, stop_tokens, start_phrases, state, prompt_options
    )
    assert result == 'f"JSONDecodeError: {l}")\n'


def test_parse_response_3():
    state = State(
        prefix='"""\r\nClean chatbot arena chat log.\r\n\r\nUsage:\r\npython3 clean_chat_data.py\r\n"""\r\nimport argparse\r\nimport datetime\r\nimport json\r\nimport os\r\nfrom pytz import timezone\r\nimport time\r\n\r\nfrom tqdm import tqdm\r\n\r\nfrom fastchat.serve.monitor.basic_stats import NUM_SERVERS\r\nfrom fastchat.serve.monitor.clean_battle_data import (\r\n    to_openai_format,\r\n    replace_model_name,\r\n)\r\nfrom fastchat.utils import detect_language\r\n\r\n\r\nNETWORK_ERROR_MSG = (\r\n    "NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.".lower()\r\n)\r\n\r\n\r\ndef get_log_files(max_num_files=None):\r\n    dates = []\r\n    for month in range(4, 12):\r\n        for day in range(1, 33):\r\n            dates.append(f"2023-{month:02d}-{day:02d}")\r\n\r\n    filenames = []\r\n    for d in dates:\r\n        for i in range(NUM_SERVERS):\r\n            name = os.path.expanduser(f"~/fastchat_logs/server{i}/{d}-conv.json")\r\n            if os.path.exists(name):\r\n                filenames.append(name)\r\n    max_num_files = max_num_files or len(filenames)\r\n    # filenames = list(reversed(filenames))\r\n    filenames = filenames[-max_num_files:]\r\n    return filenames\r\n\r\n\r\ndef clean_chat_data(log_files, action_type):\r\n    raw_data = []\r\n    for filename in tqdm(log_files, desc="read files"):\r\n        for retry in range(5):\r\n            try:\r\n                lines = open(filename).readlines()\r\n                break\r\n            except FileNotFoundError:\r\n                time.sleep(2)\r\n\r\n        for l in lines:\r\n            row = json.loads(l)\r\n            if row["type"] == action_type:\r\n                raw_data.append(row)\r\n\r\n    all_models = set()\r\n    all_ips = dict()\r\n    chats = []\r\n    ct_invalid_conv_id = 0\r\n    ct_invalid = 0\r\n    ct_network_error = 0\r\n    for row in raw_data:\r\n        try:\r\n            if action_type in ["chat", "upvote", "downvote"]:\r\n                state = row["state"]\r\n                model = row["model"]\r\n            elif action_type == "leftvote":\r\n                state = row["states"][0]\r\n                model = row["states"][0]["model_name"]\r\n            elif action_type == "rightvote":\r\n                state = row["states"][1]\r\n                model = row["states"][1]["model_name"]\r\n            conversation_id = state["conv_id"]\r\n        except KeyError:\r\n            ct_invalid_conv_id += 1\r\n            continue\r\n\r\n        if conversation_id is None:\r\n            ct_invalid_conv_id += 1\r\n            continue\r\n\r\n        conversation = to_openai_format(state["messages"][state["offset"] :])\r\n        if not isinstance(model, str):\r\n            ct_invalid += 1\r\n            continue\r\n        model = replace_model_name(model, row["tstamp"])\r\n\r\n        try:\r\n            lang_code = detect_language(state["messages"][state["offset"]][1])\r\n        except IndexError:\r\n            ct_invalid += 1\r\n            continue\r\n\r\n        if not all(isinstance(x["content"], str) for x in conversation):\r\n            ct_invalid += 1\r\n            continue\r\n\r\n        messages = "".join([x["content"] for x in conversation]).lower()\r\n        if NETWORK_ERROR_MSG in messages:\r\n            ct_network_error += 1\r\n            continue\r\n\r\n        ip = row["ip"]\r\n        if ip not in all_ips:\r\n            all_ips[ip] = len(all_ips)\r\n        user_id = all_ips[ip]\r\n\r\n        chats.append(\r\n            dict(\r\n                conversation_id=conversation_id,\r\n                model=model,\r\n                conversation=conversation,\r\n                turn=len(conversation) // 2,\r\n                language=lang_code,\r\n                user_id=user_id,\r\n                tstamp=row["tstamp"],\r\n            )\r\n        )\r\n\r\n        all_models.update([model])\r\n\r\n    chats.sort(key=lambda x: x["tstamp"])\r\n    last_updated_tstamp = chats[-1]["tstamp"]\r\n    last_updated_datetime = datetime.datetime.fromtimestamp(\r\n        last_updated_tstamp, tz=timezone("US/Pacific")\r\n    ).strftime("%Y-%m-%d %H:%M:%S %Z")\r\n\r\n    # Deduplication\r\n    dedup_chats = []\r\n    visited_conv_ids = set()\r\n    for i in reversed(range(len(chats))):\r\n        if chats[i]["conversation_id"] in visited_conv_ids:\r\n            continue\r\n        visited_conv_ids.add(chats[i]["conversation_id"])\r\n        dedup_chats.append(chats[i])\r\n\r\n    print(\r\n        f"#raw: {len(raw_data)}, #chat: {len(chats)}, #dedup_chat: {len(dedup_chats)}"\r\n    )\r\n    print(\r\n        f"#invalid_conv_id: {ct_invalid_conv_id}, #network_error: {ct_network_error}, #invalid: {ct_invalid}"\r\n    )\r\n    print(f"#models: {len(all_models)}, {all_models}")\r\n    print(f"last-updated: {last_updated_datetime}")\r\n\r\n    return list(reversed(dedup_chats))\r\n\r\n\r\nif __name__ == "__main__":\r\n    parser = argparse.ArgumentParser()\r\n    parser.add_argument("--action-type", type=str, default="chat")\r\n    parser.add_argument("--max-num-files", type=int)\r\n    parser.add_argument("--vision',
        suffix='\r\n    args = parser.parse_args()\r\n\r\n    log_files = get_log_files(args.max_num_files)\r\n    chats = clean_chat_data(log_files, args.action_type)\r\n    last_updated_tstamp = chats[-1]["tstamp"]\r\n    cutoff_date = datetime.datetime.fromtimestamp(\r\n        last_updated_tstamp, tz=timezone("US/Pacific")\r\n    ).strftime("%Y%m%d")\r\n\r\n    output = f"clean_{args.action_type}_conv_{cutoff_date}.json"\r\n    with open(output, "w") as fout:\r\n        json.dump(chats, fout, indent=2, ensure_ascii=False)\r\n    print(f"Write cleaned data to {output}")\r\n',
    )
    completion = """, type=str, default="False")
    args = parser.parse_args()

    log_files = get_log_files(args.max_num_files)
    chats = clean_chat_data(log_files, args.action_type)
    last_updated_tstamp = chats[-1]["tstamp"]"""
    max_lines = 10
    stop_tokens = []
    start_phrases = []
    prompt_options = ["overlap"]

    result = parse_response(
        completion, max_lines, stop_tokens, start_phrases, state, prompt_options
    )

    assert result == ', type=str, default="False")'


def test_parse_response_max_lines():
    completion = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    max_lines = 2
    stop_tokens = []
    start_phrases = []
    state = State(prefix="", suffix="")

    result = parse_response(completion, max_lines, stop_tokens, start_phrases, state)
    assert result == "Line 1\nLine 2\nLine 3"


def test_parse_response_multiple_start_phrases():
    completion = "Ignore this\nStart1\nLine 1\nLine 2\nStop"
    max_lines = 10
    stop_tokens = ["Stop"]
    start_phrases = ["Start2", "Start1"]
    state = State(prefix="", suffix="")

    result = parse_response(completion, max_lines, stop_tokens, start_phrases, state)
    assert result == "\nLine 1\nLine 2\n"


def test_parse_response_multiple_stop_tokens():
    completion = "Start\nLine 1\nLine 2\nStop1\nLine 3\nStop2"
    max_lines = 10
    stop_tokens = ["Stop1", "Stop2"]
    start_phrases = ["Start"]
    state = State(prefix="", suffix="")

    result = parse_response(completion, max_lines, stop_tokens, start_phrases, state)
    assert result == "\nLine 1\nLine 2\n"


def test_parse_response_escaped_newlines_in_string():
    completion = 'print("Line 1\\nLine 2\\nLine 3")'
    max_lines = 10
    stop_tokens = []
    start_phrases = []
    state = State(prefix="", suffix="")

    result = parse_response(completion, max_lines, stop_tokens, start_phrases, state)
    assert result == 'print("Line 1\\nLine 2\\nLine 3")'
