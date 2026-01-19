import sys
sys.path.append("./")

from libero.libero import benchmark
import json
import torch

result_success_dict = {
        'Objects Layout': 0,
        'Language Instructions': 0,
        'Light Conditions': 0,
        'Camera Viewpoints': 0,
        'Robot Initial States' : 0,
        'Background Textures': 0,
        'Sensor Noise': 0,
    }

result_fail_dict = {
        'Objects Layout': 0,
        'Language Instructions': 0,
        'Light Conditions': 0,
        'Camera Viewpoints': 0,
        'Robot Initial States' : 0,
        'Background Textures': 0,
        'Sensor Noise': 0,
    }

task_suite_names = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10']
with open("name_to_category.json", "r") as f:
    name_to_category =  json.load(f)

for task_suite_name in task_suite_names:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    lang_2_name = dict()
    for task_id in range(task_suite.n_tasks):
        lang_2_name[task_suite.get_task(task_id).language] = task_suite.get_task_names()[task_id]

    with open(task_suite_name+'_outcome.json', "r") as f:
        task_suite_name_outcome =  json.load(f)

    for lang, success in task_suite_name_outcome.items():
        task_name = lang_2_name[lang]
        category = name_to_category[task_name]
        if success:
            result_success_dict[category] += 1
        else:
            result_fail_dict[category] += 1

    print('a')
success_rate_dict = {
    'Objects Layout': result_success_dict['Objects Layout'] / (result_success_dict['Objects Layout'] + result_fail_dict['Objects Layout']),
    'Language Instructions': result_success_dict['Language Instructions'] / (result_success_dict['Language Instructions'] + result_fail_dict['Language Instructions']),
    'Light Conditions': result_success_dict['Light Conditions'] / (result_success_dict['Light Conditions'] + result_fail_dict['Light Conditions']),
    'Camera Viewpoints': result_success_dict['Camera Viewpoints'] / (result_success_dict['Camera Viewpoints'] + result_fail_dict['Camera Viewpoints']),
    'Robot Initial States': result_success_dict['Robot Initial States'] / (result_success_dict['Robot Initial States'] + result_fail_dict['Robot Initial States']),
    'Background Textures': result_success_dict['Background Textures'] / (result_success_dict['Background Textures'] + result_fail_dict['Background Textures']),
    'Sensor Noise': result_success_dict['Sensor Noise'] / (result_success_dict['Sensor Noise'] + result_fail_dict['Sensor Noise']),
}
print(success_rate_dict)