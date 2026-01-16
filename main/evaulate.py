import re
import string
import collections
import numpy as np
import json  # Import for JSON handling

def normalize_answer(s):
    """Normalize the answer: lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    """Get tokens from a normalized answer"""
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    """Compute exact match (EM) score"""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    """Compute F1 score"""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is empty, return 1 if both are empty, else 0
        return int(gold_toks == pred_toks)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def parse_qa_file(file_path):
    """Parse QA file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    qa_blocks = content.strip().split('---\n')
    qa_pairs = []

    for block in qa_blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')
        qa_pair = {}

        for line in lines:
            if line.startswith('qid:'):
                qa_pair['qid'] = line.replace('qid:', '').strip()
            elif line.startswith('question:'):
                qa_pair['question'] = line.replace('question:', '').strip()
            elif line.startswith('predicted_answer:'):
                qa_pair['predicted_answer'] = line.replace('predicted_answer:', '').strip()
            elif line.startswith('golden_answers:'):
                golden_answers_text = line.replace('golden_answers:', '').strip()
                try:
                    qa_pair['golden_answers'] = json.loads(golden_answers_text)
                except json.JSONDecodeError:
                    if ',' in golden_answers_text:
                        qa_pair['golden_answers'] = [item.strip() for item in golden_answers_text.split(',')]
                    else:
                        qa_pair['golden_answers'] = [golden_answers_text]

        if len(qa_pair) == 4:
            qa_pairs.append(qa_pair)

    return qa_pairs

def evaluate(file_path):
    """Evaluate EM and F1 scores"""
    qa_pairs = parse_qa_file(file_path)

    em_scores = []
    f1_scores = []
    results = []

    for qa in qa_pairs:
        qid = qa['qid']
        question = qa['question']
        pred = qa['predicted_answer']
        gold_answers = qa['golden_answers']

        max_em = 0
        max_f1 = 0

        for gold in gold_answers:
            em = compute_exact(gold, pred)
            f1 = compute_f1(gold, pred)

            max_em = max(max_em, em)
            max_f1 = max(max_f1, f1)

        em_scores.append(max_em)
        f1_scores.append(max_f1)

        results.append({
            'qid': qid,
            'question': question,
            'predicted': pred,
            'golden': gold_answers,
            'em': max_em,
            'f1': max_f1
        })

    avg_em = np.mean(em_scores) if em_scores else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0

    return {
        'results': results,
        'avg_em': avg_em,
        'avg_f1': avg_f1
    }

def print_results(eval_results):
    """Print evaluation results"""
    print("Detailed Evaluation Results:")
    print("-" * 80)

    for idx, result in enumerate(eval_results['results'], 1):
        print(f"Question {idx}:")
        print(f"QID: {result['qid']}")
        print(f"Question: {result['question']}")
        print(f"Predicted Answer: {result['predicted']}")
        print(f"Golden Answers: {result['golden']}")
        print(f"EM: {result['em']}")
        print(f"F1: {result['f1']:.4f}")
        print("-" * 80)

    print("Overall Scores:")
    print(f"Average EM: {eval_results['avg_em']:.4f} ({eval_results['avg_em']*100:.2f}%)")
    print(f"Average F1: {eval_results['avg_f1']:.4f} ({eval_results['avg_f1']*100:.2f}%)")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python em_f1_evaluation.py <qa_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    eval_results = evaluate(file_path)
    print_results(eval_results)
