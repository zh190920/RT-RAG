import json
import random
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from tree_decompose import decompose_and_answer_with_variants
import aiofiles
import config  # Import config.py

# Set random seed for reproducibility
random.seed(42)

# Load configuration from config.py
dataset_name = config.DATASET
data_path = config.DATA_PATH
output_dir_root = config.OUTPUT_DIR_ROOT
max_concurrent = config.MAX_CONCURRENT
chunk_size = config.CHUNK_SIZE
topk1 = config.TOPK1
topk2 = config.TOPK2
method = config.METHOD

# Dynamically construct output directory
output_dir = os.path.join(output_dir_root, dataset_name, f"{method}_chunk{chunk_size}_topk1_{topk1}_topk2_{topk2}")

# Determine next available file name (1.txt, 2.txt, 3.txt, etc.)
def get_next_available_file(output_dir):
    base_file_name = os.path.join(output_dir, "1.txt")
    if not os.path.exists(base_file_name):
        return base_file_name

    file_num = 2
    while os.path.exists(os.path.join(output_dir, f"{file_num}.txt")):
        file_num += 1
    return os.path.join(output_dir, f"{file_num}.txt")

# Get the next available file path
output_file_path = get_next_available_file(output_dir)

# Semaphore to limit concurrency
semaphore = None  # Will be initialized in the main function

# Improved asynchronous file writing function
async def write_result_to_file(result, file_path):
    async with aiofiles.open(file_path, 'a', encoding='utf-8') as fout:
        if result["success"]:
            await fout.write(f"qid: {result['qid']}\n")
            await fout.write(f"question: {result['question']}\n")
            await fout.write(f"predicted_answer: {result['predicted_answer']}\n")

            if isinstance(result['golden_answers'], str):
                if ',' in result['golden_answers']:
                    golden_answers = [item.strip() for item in result['golden_answers'].split(',')]
                else:
                    golden_answers = [result['golden_answers']]
            elif not isinstance(result['golden_answers'], list):
                golden_answers = [str(result['golden_answers'])]
            else:
                golden_answers = result['golden_answers']

            await fout.write(f"golden_answers: {json.dumps(golden_answers, ensure_ascii=False)}\n")
        else:
            await fout.write(f"qid: {result['qid']}\n")
            await fout.write(f"error: {result['error']}\n")
        await fout.write("---\n")
        await fout.flush()

# Asynchronous function to process a single example
async def process_example(example, idx):
    async with semaphore:
        try:
            qid = example["_id"] if "_id" in example else f"unknown_id_{idx}"
            question = example["input"]
            golden_answers = example["answers"]

            loop = asyncio.get_event_loop()
            try:
                predicted_answer = await loop.run_in_executor(
                    None,
                    lambda: decompose_and_answer_with_variants(question=question)
                )
            except Exception as e:
                predicted_answer = f"Error: {str(e)}"

            if "Error" in predicted_answer:
                print(f"Detected error: {predicted_answer}, retrying question: {qid}")
                return await process_example(example, idx)  # Retry recursively

            result = {
                "qid": qid,
                "question": question,
                "predicted_answer": predicted_answer,
                "golden_answers": golden_answers,
                "idx": idx,
                "success": True
            }

            await write_result_to_file(result, output_file_path)

            return result

        except Exception as e:
            error_result = {
                "qid": f"error_{idx}",
                "error": f"Error during processing: {str(e)}",
                "idx": idx,
                "success": False
            }

            await write_result_to_file(error_result, output_file_path)

            return error_result

async def main():
    global semaphore

    semaphore = asyncio.Semaphore(max_concurrent)

    print(f"Loading dataset: {dataset_name}")
    data = load_dataset('json', data_files=data_path, split='train')
    print(f"Successfully loaded dataset with {len(data)} examples")

    os.makedirs(output_dir, exist_ok=True)

    tasks = [process_example(data[idx], idx) for idx in range(len(data))]

    results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {dataset_name} dataset")

    success_count = sum(1 for r in results if r.get("success", False))
    fail_count = len(results) - success_count

    print(f"Processing completed. Main results saved to {output_file_path}")
    print(f"Total examples processed: {len(results)}, Success: {success_count}, Failures: {fail_count}")

if __name__ == "__main__":
    asyncio.run(main())
