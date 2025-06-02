from typing import Optional
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from .settings import AppSettings
from .utils import summary


def main(
        settings,
        ifname,
        ofname: Optional[str] = None,
        quiet: bool = False,
        debug: bool = False,
        retry: int = 3,
        retry_delay: int = 1,
        ):
    import json
    from topic_classification.llm import GeminiAPILLM
    from topic_classification.classify_topics import classify_topics

    llm = GeminiAPILLM(settings.gemini_api_key, retry=retry, retry_delay=retry_delay)

    try:
        with open(ifname, 'r') as fp:
            text_list = json.load(fp)
    except:
        print(f'{ifname} is not a valid json file. exit.')
        return
    else:
        if not isinstance(text_list, list):
            print('Invalid json format (should be an array of str). exit')
            return

    pred_dict = classify_topics(
        llm, text_list,
        other_key="其他",
        sleep=retry_delay,
        batch_size=settings.batch_size,
        retry=retry,
        retry_delay=retry_delay,
        quiet=quiet,
        debug=debug,
    )

    summary(pred_dict)

    if ofname is not None:
        with open(ofname, 'w') as fp:
            json.dump(pred_dict, fp, ensure_ascii=False)


def cli():
    parser = ArgumentParser(
        prog='classify-topics',
        description='Classify topics of the given sentences according to the sentences and make a summary.',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-o', '--output',
        required=False,
        help='output json path',
    )
    parser.add_argument(
        '--retry',
        type=int,
        default=3,
        help='maximum retry counts'
    )
    parser.add_argument(
        '--retry-delay',
        type=float,
        default=1,
        help='time interval between each call to llm'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='don\'t print any processing message'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='debug mode',
    )
    parser.add_argument(
        'file',
        help='input json file',
    )

    args = parser.parse_args()

    settings = AppSettings()
    main(
        settings,
        args.file,
        ofname=args.output,
        quiet=args.quiet,
        debug=args.debug,
        retry=args.retry,
        retry_delay=args.retry_delay,
    )

