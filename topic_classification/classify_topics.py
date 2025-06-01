import json
import math
import time
from typing import Optional, List, Dict

from .llm import BaseLLM
from .exceptions import MaximumRetryError, CustomError


def get_prompt(
        text_list: List[str],
        classes: Optional[List[str]] = None,
        ):
    text_list_json = json.dumps(text_list, indent=4, ensure_ascii=False)
    json_example = json.dumps({
        "類別文字1": [
            "李昂真的太帥了！這代的建模跟聲音都超有魅力😍",
            "打教團的時候真的是邊打邊罵，太變態了啦那群人"
        ],
        "類別文字2": [
            "Steam壓倒性好評不是沒原因，Capcom這次有回春"
        ]
    }, indent=4, ensure_ascii=False)
    if not classes:
        s = f'''請將下列陣列中的文句根據話題分成多個類別，並將結果以 JSON 格式輸出，請確認每個文句皆有出現在輸出結果中。 輸出範例：\n{json_example}\n文句陣列為：\n{text_list_json}
        '''
    else:
        classes_str = '、'.join([f'"{c}"' for c in classes[:-1]])
        classes_str += f'以及"{classes[-1]}"'
        s = f'''請將以下陣列中的文句根據話題分為以下類別：{classes_str}，並將結果以 JSON 格式輸出，請確認每個文句皆有出現在輸出結果中。輸出範例：\n{json_example}\n文句陣列為：\n{text_list_json}
        '''
    return s



def merge_pred_dict(pred_dict: Dict[str, List[str]], other: Dict[str, List[str]]):
    for k, v in other.items():
        if k not in pred_dict:
            pred_dict[k] = []
        pred_dict[k].extend(v)
    return pred_dict


def _classify_topics(
        llm: BaseLLM,
        text_list: List[str],
        other_key: str = "其他",
        batch_size: int = 100,
        sleep: int = 1,
        retry: int = 3,
        retry_delay: float = 1,
        quiet: bool = False,
        debug: bool = False,
        ):
    classes = []
    pred_dict = {} # {<cls>: [<text>]}
    N = len(text_list)
    i = 0
    retry_cnt = 0
    excs = []
    while (i < N):
        if not quiet:
            print(f'[{i//batch_size+1}/{math.ceil(N/batch_size)}] process {i}-{i+batch_size}')
        if len(classes) > 0 and other_key not in classes:
            classes.append(other_key)
        _text_list = text_list[i:min(i+batch_size, N)]
        prompt = get_prompt(_text_list, classes=classes)
        orig = llm(prompt).strip()
        """
        ```json
        {
            ...
        }
        ```
        """
        pred = orig
        if pred.startswith('```json'):
            pred = pred[7:]
        if pred.endswith('```'):
            pred = pred[:-3]
        pred = pred.strip()
        try:
            _pred_dict = json.loads(pred) # {<cls>: [<text>]}
        except:
            if retry_cnt >= retry:
                raise MaximumRetryError(excs)
            if not quiet:
                if debug:
                    print(orig)
                print('Json loads fail, retry')
            retry_cnt += 1
            excs.append(CustomError('Fail to load Json'))
            time.sleep(retry_delay)
            continue
        else:
            if sum([len(v) for v in _pred_dict.values()]) != len(_text_list):
                if retry_cnt >= retry:
                    raise MaximumRetryError(excs)
                if not quiet:
                    if debug:
                        print(orig)
                    print('Predict size mismatch, retry')
                retry_cnt += 1
                excs.append(CustomError('Predict size mismatch'))
                time.sleep(retry_delay)
                continue
            else:
                i += batch_size
        classes = list(_pred_dict.keys())
        
        merge_pred_dict(pred_dict, _pred_dict)
        time.sleep(sleep)
    return pred_dict


def classify_topics(
        llm: BaseLLM,
        text_list: List[str],
        other_key: str = "其他",
        batch_size: int = 100,
        sleep: int = 1,
        max_iteration: Optional[int] = None,
        retry: int = 3,
        retry_delay: float = 1,
        quiet: bool = False,
        debug: bool = False,
        ):
    pred_dict = {}
    _text_list = text_list
    it = 1
    while True:
        if not quiet:
            print(f'Iteration {it} ({len(_text_list)})')
        _pred_dict = _classify_topics(llm, _text_list, other_key=other_key, sleep=sleep, batch_size=batch_size, retry=retry, retry_delay=retry_delay)
        tmp = _pred_dict.get(other_key, [])
        if len(tmp) == 0 or len(tmp) == len(text_list):
            break
        if max_iteration is not None and it >= max_iteration:
            break
        _text_list = tmp
        merge_pred_dict(pred_dict, _pred_dict)
        it += 1

    _pred_dict = pred_dict
    pred_dict = {other_key: []}
    for k, v in _pred_dict.items():
        if len(v) <= 1:
            pred_dict[other_key].extend(v)
        else:
            pred_dict[k] = v
    return pred_dict
