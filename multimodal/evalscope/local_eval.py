import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from evalscope.run import run_task
from evalscope.summarizer import Summarizer

from evalscope import TaskConfig


task_cfg_dict = TaskConfig(
    work_dir='outputs',
    eval_backend='VLMEvalKit',
    eval_config=
        {'data': ['RealWorldQA'],
        'mode': 'all',
        'model': [ 
            {'name': 'Qwen2-VL-7B-Instruct',
            'model_path': '/home/longchen/llm/model/Qwen2-VL-7B-Instruct',
            'max_new_tokens': 100,
            }
          ],
        'reuse': False,
        'nproc': 8,
        'judge': 'exact_matching'}
 )

def run_eval():
    # 选项 1: python 字典
    task_cfg = task_cfg_dict

    run_task(task_cfg=task_cfg)

    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>> The report list: {report_list}')

run_eval()