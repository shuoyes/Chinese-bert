import re

def find_max_metrics(file_name, start_line):
    max_c_acc = float('-inf')
    max_acc_adj = float('-inf')
    max_precision = float('-inf')
    max_recall = float('-inf')
    max_f1 = float('-inf')
    max_qwk = float('-inf')

    with open(file_name, 'r') as file:
        data = file.readlines()

        # Start reading from the specified line
    for line in data[start_line:]:
        if not line.rstrip().startswith("test"):
            continue
        train_metrics = re.split(r'\s{3,}', line.strip())
        # train_metrics = line.split("     ")
        # 创建字典
        metrics_dict = {}
        for metric in train_metrics[1:]:  # 跳过第一个词'test'
            key, value = metric.split(": ")
            metrics_dict[key] = value[:-1]
        c_acc = float(metrics_dict["C_Acc"])
        acc_adj = float(metrics_dict["Acc_adj"])
        precision = float(metrics_dict["Precision"])
        recall = float(metrics_dict["Recall"])
        f1 = float(metrics_dict["F1"])
        qwk = float(metrics_dict["QWK"])

        if c_acc > max_c_acc:
            max_c_acc = c_acc
        if acc_adj > max_acc_adj:
            max_acc_adj = acc_adj
        if precision > max_precision:
            max_precision = precision
        if recall > max_recall:
            max_recall = recall
        if f1 > max_f1:
            max_f1 = f1
        if qwk > max_qwk:
            max_qwk = qwk

    return max_c_acc, max_acc_adj, max_precision, max_recall, max_f1, max_qwk


file_name = r'C:\Users\shuoye\Desktop\experiment_results\DTRAtraining_log_CMER(12)-30.txt'  # Specify the file name here or pass it as an argument to the function
start_line = 335  # Specify the starting line here
max_metrics = find_max_metrics(file_name, start_line)
print(f"Max train C_Acc: {max_metrics[0]:.3f}")
print(f"Max train Acc_adj: {max_metrics[1]:.3f}")
print(f"Max train Precision: {max_metrics[2]:.3f}")
print(f"Max train Recall: {max_metrics[3]:.3f}")
print(f"Max train F1: {max_metrics[4]:.3f}")
print(f"Max train QWK: {max_metrics[5]:.3f}")