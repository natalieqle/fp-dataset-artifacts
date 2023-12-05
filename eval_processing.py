import csv
import json

def preprocess_lit():
    rows = []
    with open('/Users/Natalie/Desktop/Final_Project_NLP/fp-dataset-artifacts/data/all_eval_data.csv', 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        skipped_original = 0
        num_entailed = 0
        num_neutral = 0
        num_contr = 0
        # label 0 is "entailed", 1 is "neutral", and 2 is "contradiction"
        for row in csvreader:
            if row[0] == 'original':
                skipped_original += 1
                continue
            label = row[3]
            if label == 'entailment':
                row[3] = 0
                num_entailed += 1
            elif label == 'neutral':
                row[3] = 1
                num_neutral += 1
            elif label == 'contradiction':
                row[3] = 2
                num_contr += 1
            else:
                continue
            rows.append(row)
    print(f'skipped {skipped_original} examples, total is {len(rows)}')
    print(f'entailed: {num_entailed}, neutral: {num_neutral}, contradiction: {num_contr}')

    with open("/Users/Natalie/Downloads/lit_only_2.json", "w") as outfile:
        for row in rows:
            dictionary = {
                "caption": row[0],
                "premise": row[1],
                "hypothesis": row[2],
                "label": row[3]
            }
            json.dump(dictionary, outfile)
            outfile.write("\n")

def eval_lit_only(eval_csv: str, output_prefix: str):
    base = '/Users/Natalie/Desktop/Final_Project_NLP/fp-dataset-artifacts/'
    dev_path = base + 'minimax_trained_eval/no_grad_256/final/lit_only/'

    caption_map = {}
    with open(base + 'data/all_eval_data.csv', 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            if row[0] == 'original':
                continue
            caption_map[f'{row[1]}{row[2]}'] = row[0]

    columns = ['premise', 'hypothesis', 'gold', 'dev_pred', 'processed_caption', 'caption']
    rows = []

    with open(dev_path + eval_csv, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        may = 0
        it_cleft = 0
        future_simple = 0
        past_simple = 0
        passive = 0

        for row in csvreader:
            caption = caption_map[f'{row[0]}{row[1]}']
            processed_caption = ''
            if "may" in caption:
                may += 1
                processed_caption += "MAY."
            if "it cleft" in caption:
                it_cleft += 1
                processed_caption += "IT CLEFT."
            if "future simple" in caption:
                future_simple += 1
                processed_caption += "FUTURE SIMPLE."
            if "past simple" in caption:
                past_simple += 1
                processed_caption += "PAST SIMPLE."
            if "passive" in caption:
                passive += 1
                processed_caption += "PASSIVE."
            rows.append([row[0], row[1], row[2], row[3], processed_caption, caption])


    with open(dev_path + output_prefix + '_eval.csv', 'w') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(columns)
        csvwriter.writerows(rows)

    for i in range(3):
        print(rows[i])

    print(f'may: {may}, it cleft: {it_cleft}, future simple: {future_simple}, past simple: {past_simple}, passive: {passive}')

def compare(pretrained_path: str, dev_path: str, binary: bool):
    base = '/Users/Natalie/Desktop/Final_Project_NLP/fp-dataset-artifacts/'
    pretrained_preds = open(base + pretrained_path + 'eval_predictions.jsonl').read().splitlines()
    dev_preds = open(base + dev_path + 'eval_predictions.jsonl').read().splitlines()

    label_map = {}
    label_map[0] = 'entailment'
    if binary:
        label_map[1] = 'not entailed'
        label_map[2] = 'not entailed'
    else:
        label_map[1] = 'neutral'
        label_map[2] = 'contradiction'

    degraded_rows = []
    improved_rows = []
    columns = ['premise', 'hypothesis', 'gold', 'dev_pred', 'pretrained_pred']

    if binary:
        columns.append('heuristic')

    if 'anli' in pretrained_path:
        columns.append('reason')

    pretained_res = {}
    pretrained_pred_label = {}

    num_pretrained_lex = 0
    num_pretrained_const = 0
    num_pretrained_sub = 0

    num_dev_lex = 0
    num_dev_const = 0
    num_dev_sub = 0

    pretrained_may = 0
    pretrained_it_cleft = 0
    pretrained_future_simple = 0
    pretrained_past_simple = 0
    pretrained_passive = 0

    dev_may = 0
    dev_it_cleft = 0
    dev_future_simple = 0
    dev_past_simple = 0
    dev_passive = 0

    gold_entailed_but_dev_neutral = 0
    gold_entailed_but_dev_contradiction = 0
    gold_neutral_but_dev_entailed = 0
    gold_neutral_but_dev_contradiction = 0
    gold_contradiction_but_dev_entailed = 0
    gold_contradiction_but_dev_neutral = 0

    false_positive = 0
    false_negative = 0

    for pred in pretrained_preds:
        p = json.loads(pred)
        gold_label = label_map[p["label"]]
        pred_label = label_map[p["predicted_label"]]
        key = f'{p["premise"]}{p["hypothesis"]}'
        pretrained_pred_label[key] = pred_label
        if gold_label == pred_label:
            pretained_res[key] = 'CORRECT'
        else:
            pretained_res[key] = 'INCORRECT'
            if gold_label == "contradiction":
                if pred_label == "entailment":
                    gold_contradiction_but_dev_entailed += 1
                else:
                    gold_contradiction_but_dev_neutral += 1
            if gold_label == "entailment":
                if pred_label == "contradiction":
                    gold_entailed_but_dev_contradiction += 1
                else:
                    gold_entailed_but_dev_neutral += 1
            if gold_label == "neutral":
                if pred_label == "entailment":
                    gold_neutral_but_dev_entailed += 1
                else:
                    gold_neutral_but_dev_contradiction += 1
            if "heuristic" in p.keys():
                heur = p["heuristic"]
                if heur == 'lexical_overlap':
                    num_pretrained_lex += 1
                elif heur == 'constituent':
                    num_pretrained_const += 1
                elif heur == 'subsequence':
                    num_pretrained_sub += 1
            if "caption" in p.keys():
                caption = p["caption"]
                if "may" in caption:
                    pretrained_may += 1
                if "it cleft" in caption:
                    pretrained_it_cleft += 1
                if "future simple" in caption:
                    pretrained_future_simple += 1
                if "past simple" in caption:
                    pretrained_past_simple += 1
                if "passive" in caption:
                    pretrained_passive += 1

    print('PRETRAINED')
    print(
        f'gold_entailed_but_dev_contradiction: {gold_entailed_but_dev_contradiction}, gold_entailed_but_dev_neutral: {gold_entailed_but_dev_neutral}')
    print(
        f'gold_neutral_but_dev_contradiction: {gold_neutral_but_dev_contradiction}, gold_neutral_but_dev_entailed: {gold_neutral_but_dev_entailed}')
    print(
        f'gold_contradiction_but_dev_neutral: {gold_contradiction_but_dev_neutral}, gold_contradiction_but_dev_entailed: {gold_contradiction_but_dev_entailed}')

    gold_entailed_but_dev_neutral = 0
    gold_entailed_but_dev_contradiction = 0
    gold_neutral_but_dev_entailed = 0
    gold_neutral_but_dev_contradiction = 0
    gold_contradiction_but_dev_entailed = 0
    gold_contradiction_but_dev_neutral = 0

    for pred in dev_preds:
        p = json.loads(pred)
        gold_label = label_map[p["label"]]
        pred_label = label_map[p["predicted_label"]]

        if gold_label == pred_label:
            dev_res = 'CORRECT'
        else:
            dev_res = 'INCORRECT'
            if pred_label == 'not entailed':
                false_negative += 1
            else:
                false_positive += 1
            if gold_label == "contradiction":
                if pred_label == "entailment":
                    gold_contradiction_but_dev_entailed += 1
                else:
                    gold_contradiction_but_dev_neutral += 1
            if gold_label == "entailment":
                if pred_label == "contradiction":
                    gold_entailed_but_dev_contradiction += 1
                else:
                    gold_entailed_but_dev_neutral += 1
            if gold_label == "neutral":
                if pred_label == "entailment":
                    gold_neutral_but_dev_entailed += 1
                else:
                    gold_neutral_but_dev_contradiction += 1
            if "heuristic" in p.keys():
                heur = p["heuristic"]
                if heur == 'lexical_overlap':
                    num_dev_lex += 1
                elif heur == 'constituent':
                    num_dev_const += 1
                elif heur == 'subsequence':
                    num_dev_sub += 1
            if "caption" in p.keys():
                caption = p["caption"]
                if "may" in caption:
                    dev_may += 1
                if "it cleft" in caption:
                    dev_it_cleft += 1
                if "future simple" in caption:
                    dev_future_simple += 1
                if "past simple" in caption:
                    dev_past_simple += 1
                if "passive" in caption:
                    dev_passive += 1

        key = f'{p["premise"]}{p["hypothesis"]}'
        pt_res = pretained_res[key]
        if dev_res != pt_res:
            row = [p["premise"], p["hypothesis"], gold_label, pred_label, pretrained_pred_label[key]]
            if "heuristic" in p.keys():
                row.append(p["heuristic"])
            if 'anli' in pretrained_path:
                row.append(p["reason"])

            if dev_res == 'CORRECT':
                improved_rows.append(row)
            else:
                degraded_rows.append(row)

        output_prefix = 'binary_' if binary else 'multi_'
        with open(base + dev_path + output_prefix + 'improved.csv', 'w') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(columns)
            csvwriter.writerows(improved_rows)

        with open(base + dev_path + output_prefix + 'degraded.csv', 'w') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(columns)
            csvwriter.writerows(degraded_rows)

    print('DEV')
    print(f'false positive: {false_positive}, false negative: {false_negative}')
    print(
        f'gold_entailed_but_dev_contradiction: {gold_entailed_but_dev_contradiction}, gold_entailed_but_dev_neutral: {gold_entailed_but_dev_neutral}')
    print(
        f'gold_neutral_but_dev_contradiction: {gold_neutral_but_dev_contradiction}, gold_neutral_but_dev_entailed: {gold_neutral_but_dev_entailed}')
    print(
        f'gold_contradiction_but_dev_neutral: {gold_contradiction_but_dev_neutral}, gold_contradiction_but_dev_entailed: {gold_contradiction_but_dev_entailed}')

    print(f'PRETRAINED. lex: {num_pretrained_lex}, sub: {num_pretrained_sub}, const: {num_pretrained_const}')
    print(f'DEV. lex: {num_dev_lex}, sub: {num_dev_sub}, const: {num_dev_const}')

    print(f'PRETRAINED. may: {pretrained_may}, it cleft: {pretrained_it_cleft}, future simple: {pretrained_future_simple}, past simple: {pretrained_past_simple}, passive: {pretrained_passive}')
    print(f'DEV. may: {dev_may}, it cleft: {dev_it_cleft}, future simple: {dev_future_simple}, past simple: {dev_past_simple}, passive: {dev_passive}')

    print(f'stats for: {pretrained_path}')
    num_improved = len(improved_rows)
    print(f'improved: {num_improved}')
    for i in range(3):
        if i < num_improved:
            print(improved_rows[i])

    num_degraded = len(degraded_rows)
    print(f'degraded: {num_degraded}')
    for i in range(3):
        if i < num_degraded:
            print(degraded_rows[i])

if __name__ == "__main__":
    # preprocess_lit();
    compare(pretrained_path='pretrained_eval/snli/', dev_path='minimax_trained_eval/no_grad_256/final/snli/', binary=False)
    compare(pretrained_path='pretrained_eval/hans/', dev_path='minimax_trained_eval/no_grad_256/final/hans/', binary=True)
    compare(pretrained_path='pretrained_eval/lit_only/', dev_path='minimax_trained_eval/no_grad_256/final/lit_only/', binary=False)
    compare(pretrained_path='pretrained_eval/anli/test_r3/', dev_path='minimax_trained_eval/no_grad_256/final/anli/test_r3/', binary=False)
    eval_lit_only(eval_csv="multi_improved.csv", output_prefix="improved")
    eval_lit_only(eval_csv="multi_degraded.csv", output_prefix="degraded")