import sys
sys.path.append("./")
import json


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


for task_suite_name in task_suite_names:
    with open(task_suite_name+'_success_outcome.json', "r") as f:
        category_success =  json.load(f)
    with open(task_suite_name+'_fail_outcome.json', "r") as f:
        category_fail =  json.load(f)
    for category, success_count in category_success.items():
        result_success_dict[category] += success_count
    for category, fail_count in category_fail.items():
        result_fail_dict[category] += fail_count
    
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


