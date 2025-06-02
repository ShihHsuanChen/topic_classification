from typing import Dict, List


def summary(pred_dict: Dict[str, List[str]]):
    import pandas as pd
    stat = [{'class': k, 'quantity': len(v)} for k, v in pred_dict.items()]
    df = pd.DataFrame.from_records(stat)
    df.sort_values('quantity', ascending=False, inplace=True)
    df.index = list(range(len(df)))

    N = df['quantity'].sum()
    df['percentage'] = (df['quantity']/N*100).apply(lambda v: f'{v:.1f}%')
    df['quantity'] = df['quantity'].apply(lambda v: f'{v} 篇')
    df.index = [v if df.loc[v, 'class']!='其他' else len(df) for v in df.index]
    df.sort_index(inplace=True)

    df.columns = ['']*len(df.columns)
    df.index = ['']*len(df.index)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df.style.set_properties(**{'text-align': 'left'})
    print('=== 話題佔比分類結果 ===')
    print(df)

