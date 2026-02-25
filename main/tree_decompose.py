import os
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import argparse
import json
from typing import List, Tuple, Dict, Counter
from collections import Counter, defaultdict
from openai import OpenAI
from retrieve import answer_question, direct_answer

API_KEY = "00eacdd74fc047539eb47c5f2f96b916.0jZiNGZy3N9gHT0T"
BASE_URL= "https://open.bigmodel.cn/api/paas/v4"
from config import  DATASET, METHOD, CHUNK_SIZE, MIN_SENTENCE, OVERLAP, TOPK1, TOPK2, MAX_ITERATIONS, BASE_URL, API_KEY, TREES_PER_QUESTION, MAX_TOKENS,                        DECOMPOSE_TEMPERATURE, TOP_P,FREQUENCY_PENALTY, PRESENCE_PENALTY, NUM_EXAMPLES, MAX_HEIGHT, RIGHT_SUBTREE_VARIANTS, RIGHT_SUBTREE_TREES_PER_VARIANT, MAX_VARIANTS, STATS_FILE_PATH,ENHANCED_RIGHT_SUBTREE
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
import os
import datetime
from pathlib import Path

# Add this function to save tree statistics to a file
def save_tree_stats(question, answer, original_height, final_height, file_path, success=True):
    """
    Save tree statistics to a file
    
    Parameters:
    - question: The original question
    - answer: The final answer obtained
    - original_height: Initial height of the tree before expansion
    - final_height: Final height after expansion and solving
    - file_path: Path to save the stats
    - success: Whether the question was successfully answered
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Format current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Clean up the question and answer for single-line storage
    clean_question = question.replace('\n', ' ').strip()
    clean_answer = answer.replace('\n', ' ').strip() if answer else "[none]"
    success_str = "SUCCESS" if success else "FAILURE"
    
    # Prepare the statistics line
    stats_line = f"{timestamp}|{success_str}|{original_height}|{final_height}|{clean_question}|{clean_answer}\n"
    
    # Append to file
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(stats_line)
    
    print(f"Tree statistics saved to {file_path}")

def generate_response(messages, max_tokens=800, temperature=0.2, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    """Generic API call function to replace original requests.post call"""
    try:#"Qwen/Qwen2.5-14B-Instruct"
        completion = client.chat.completions.create(
            model="glm-4-flash",  # Can be changed as needed
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API call error: {str(e)}")
        return None
def generate_right_question_with_llm(parent_question, left_question, left_answer, original_right_question, max_tokens=800, temperature=0.2, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    
   
    messages = [
        {
            "role": "system",
            "content": "You are an intelligent AI assistant tasked with generating appropriate follow-up questions based on context. Generate clear, relevant questions that flow naturally from previous information."
        },
        {
            "role": "user",
            "content": f"""Based on the following information, please generate an appropriate follow-up question:

Original main question: {parent_question}
First sub-question: {left_question}
Answer to the first sub-question: {left_answer}
Original second sub-question (may contain replacement markers): {original_right_question}

Please generate a new second sub-question that:
1. Preserves all the essential information from the original second sub-question
2. Reads naturally and flows well in the conversation context
3. Replaces any markers like [answer_subquestion1] with natural language

Provide only the new question text without any additional explanation."""
        }
    ]
    
   
    response = generate_response(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    
   
    new_question = response.strip() if response else "Unable to generate a follow-up question."
    
 
    if "[answer_subquestion1]" in new_question or "[answer from" in new_question:
        new_question = new_question.replace("[answer_subquestion1]", "the aforementioned")
        new_question = new_question.replace("[answer from", "the aforementioned")
    
    return new_question

#---------------------------------------------------- Question Variant Generation -----------------------------------------------
def generate_question_variants(original_query, num_variants=2):
    """
    Generate question variants that maintain the exact semantics of the original query
    to ensure all variants would receive the same answer as the original question

    Parameters:
    original_query (str): The original query
    history_queries (list, optional): List of historical queries to avoid duplicating
    variant_type (str, optional): Type of variant to generate (e.g., 'wh_transform', 'syntax_reform', 'random')
    num_variants (int, optional): Number of unique variants to generate

    Returns:
    list: List of generated question variants that preserve exact semantics
    """
    system_prompt = """
      You are an expert at rewriting questions. Your job is to generate simpler, clearer versions of a question
    without changing its meaning or the answer it would receive.

    CRITICAL RULE: You must carefully analyze the original question to identify ANY specific constraints on the answer type.
    If the original question asks about a specific type of entity (like a city, date, person, number, etc.), 
    ALL your rewrites MUST explicitly preserve that exact entity type constraint.

    Each question you generate must:
    1. Preserve the original semantics exactly.
    2. NEVER replace a specific entity type with a more general one.
    3. Be simpler and more concise in sentence structure.
    4. Avoid unnecessary modifiers or extra descriptive phrases while retaining all semantic constraints.
    5. Use natural and direct English.

    Only return the list of revised questions, nothing else.
    """
    
    # Build user prompt with emphasis on simplification and syntax clarity
    user_prompt = f"""
    Original query: {original_query}

    
    Instructions:
    - First, identify any specific entity type being requested in the original query
    - If the original specifies a particular entity type, your rewrite MUST explicitly include that same type.
    - Do NOT generalize specific entity types under any circumstances.
    - Rewrite the question to be simpler and clearer while preserving ALL constraints.
    - All rewrites must have the EXACT SAME answer constraints as the original.
    - Do not repeat or include the original question in the outputs.
    - Return {num_variants} cleanly formatted variants, each on a new line and numbered.

    Start:
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response_text = generate_response(messages, temperature=0.6)
    
    # Parse the numbered list of variants
    variants = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('- ')):
            # Remove the numbering or bullet points
            variant = line.split('.', 1)[-1].strip() if '.' in line else line[2:].strip()
            variants.append(variant)
    
    # If no proper format is detected, treat the whole response as one variant
    if not variants and response_text.strip():
        variants = [response_text.strip()]
    
    # Return only the requested number of variants
    variants = variants[:num_variants]
    
    # Add the original query to the list of variants
    all_variants = [original_query] + variants
    
    print(f"Generated {len(variants)} question variants:")
    for i, variant in enumerate(all_variants):
        print(f"{i}. {variant}")
    
    return all_variants

def generate_right_question_with_llm(parent_question, left_question, left_answer, original_right_question, max_tokens=800, temperature=0.1, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    messages = [
        {
            "role": "system",
            "content": "You are a precise reasoning system that carefully analyzes question semantics."
        },
        {
            "role": "user",
            "content": f"""Original main question: {parent_question}
First sub-question: {left_question}
Answer to first sub-question: {left_answer}
Original second sub-question template: {original_right_question}

FIRST: Analyze whether the parent question is asking about:
- A SINGLE entity (e.g., "the person who...", "his/her")
- MULTIPLE entities (e.g., "the inventors...", "scientists who...", requires collaboration)

THEN: Generate the second sub-question following these core rules:

1. If parent question refers to A SINGLE ENTITY but answer has multiple entities:
   - Use "or" between entities (e.g., "A or B")
   - Adjust pronouns accordingly (e.g., "their" instead of "his")

2. If parent question refers to MULTIPLE ENTITIES:
   - Keep "and" between entities (e.g., "A and B")
   - Keep plural pronouns

3. Always preserve relationship structures 
Generate ONLY the replacement question:"""
        }
    ]
    
    response = generate_response(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    
    new_question = response.strip() if response else "Unable to generate a follow-up question."
    return new_question


def build_enhanced_right_subtree(original_question, left_answer, api_url=None, max_tokens=800, 
                            temperature=0.2, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, 
                            examples_db=None, num_examples=3, max_height=3,
                            num_variants=2, trees_per_variant=3):
    """
Build multiple trees for the right subtree of Sequential type, following the same logic as the main function



Parameters:

-original_question: Original right subtree problem

-left_answer: The answer of the left subtree, used to replace [answer_subquestion1]

Other parameters are the same as build_question_tree



Returns:

-best_node: The best right subtree node
    """
    global global_node_counter
    

    if "[answer_subquestion1]" in original_question:
        question = original_question.replace("[answer_subquestion1]", left_answer)
    else:
        question = original_question
    
    print(f"\n{'='*80}")
    print(f"Construct the right subtree based on the problem: '{question}'")
    print(f"{'='*80}")
    
   
    max_attempts = num_variants
    current_attempt = 0
    current_question = question
    attempted_questions = [question]  
    
    while current_attempt < max_attempts:
        print(f"\nTry #{current_attempt+1} use the question: '{current_question}'")
        
        all_trees = []
        trees_to_generate = trees_per_variant  
        
        print(f"\nGenerate a {trees_to_generate} tree (maximum height: {max_height}) for the current problem.")
        
        for j in range(trees_to_generate):
         
            tree_temp = temperature
            
            print(f"\n #{j+1} (temperature={tree_temp}, max_height={max_height}):")
            
           
            root = build_question_tree(
                current_question, api_url, max_tokens, tree_temp, top_p, frequency_penalty,
                presence_penalty, examples_db, num_examples, depth=0,
                placeholder_answers={}, max_height=max_height
            )
            
            
            height, node_count = get_tree_statistics(root)
            
           
            all_trees.append({
                'tree': root,
                'tree_num': j+1,
                'height': height,
                'node_count': node_count,
                'question_text': current_question
            })
            
            
        
       
        tree_shape_counter = Counter()
        for tree_info in all_trees:
            shape = (tree_info['height'], tree_info['node_count'])
            tree_shape_counter[shape] += 1
        
      
        print("\nTree shape frequency (height, number of nodes)")
        for shape, count in tree_shape_counter.most_common():
            print(f"height: {shape [0]}, the number of nodes: {shape [1]} - frequency: {count}")
        

        if tree_shape_counter:
            most_common_shape, _ = tree_shape_counter.most_common(1)[0]
            print(f"\nMost common shapes: Height {most_common_shape[0]}, Number of nodes {most_common_shape[1]}")
            
           
            most_common_trees = [tree_info for tree_info in all_trees 
                               if (tree_info['height'], tree_info['node_count']) == most_common_shape]
            
            if most_common_trees:
                
                best_tree_info = most_common_trees[0]
                
                
                
                return best_tree_info['tree']
        
        
        current_attempt += 1
        if current_attempt < max_attempts:
            
            
            
            new_variants = generate_question_variants(question, num_variants=1)
            
            
            if len(new_variants) > 1:
                new_question = new_variants[1]
                
                current_question = new_question
                attempted_questions.append(current_question)
            else:
                print(f"Warning: Variant generation failed. Use the original question.")
                break 
    
   
    
    leaf_node = QuestionNode(
        question=question, 
        q_type="None",
        subq1=question, 
        subq2=""
    )
    return leaf_node




#---------------------------------------------------- Question Structure Analysis -----------------------------------------------
def analyze_question_structure(question, api_url=None):
    """
    Analyze question and return its structure with limiting conditions
    """
    examples_string = (
    "Question: \"哪位毕业于斯坦福大学的女宇航员是20世纪90年代首位进行太空行走的人？\"\n"
    "CoT: Let's think step by step\n"
    "\"1. 这个问题询问的是具有多个显著特征的特定宇航员的身份。\"\n"
    "\"2. 这位宇航员被描述为女性，是斯坦福大学的毕业生，也是20世纪90年代首位进行太空行走的人。\"\n"
    "\"3. 对于已知实体，我需要识别问题中提到的明确主语。\"\n"
    "\"4. 斯坦福大学被明确提及，而20世纪90年代的太空行走是一个有特定时间限制的事件。\"\n"
    "\"5. 这位特定宇航员的身份并未直接给出——我需要确定谁符合这些标准。\"\n"
    "\"6. 由于我需要找出哪个具体的人符合这些特征，所以这位宇航员的身份还是个未知的谜团。\"\n"
    "So the structure is: [Core Query: Which astronaut Known Entities:  {Subject: 斯坦福大学, Limitation: 教育机构}, {Subject: 太空行走, Limitation: 发生在20世纪90年代} Unknown Entities: {Subject: 宇航员身份, Limitation: 女性, 斯坦福大学毕业生, 20世纪90年代首次进行太空行走}]\n\n"
    
    "Question: \"哪种既影响牲畜又影响人类的疾病，通过协调一致的疫苗接种运动，在1980年之前在全球范围内成功根除了？\"\n"
    "CoT: Let's think step by step\n"
    "\"1. 这个问题询问一种具有特定特征和历史的疾病。\"\n"
    "\"2. 这种疾病必须同时影响牲畜和人类，并且在1980年之前通过疫苗接种运动在全球范围内被根除。\"\n"
    "\"3. 对于已知实体，我需要识别问题中明确提到的主语。\"\n"
    "\"4. 牲畜和人类被明确提及为受影响的群体。\"\n"
    "\"5. 疫苗接种运动和1980年的时间范围是明确提到的参数。\"\n"
    "\"6. 我需要查明的是具体的疾病种类，这使其成为一个未知的实体。\"\n"
    "So the structure is: [Core Query: What disease Known Entities: {Subject: 牲畜, Limitation: 受这种疾病影响}, {Subject: 人类, Limitation: 受这种疾病影响}, {Subject: 疫苗接种活动, Limitation: 协调一致，于1980年完成，覆盖全球范围} Unknown Entities: {Subject: 疾病识别, Limitation: 既影响了牲畜也影响了人类，于1980年被根除}]\n\n"
    
    "Question: \"那位建造了巴塞罗那著名大教堂的建筑师所设计的建筑，都共同采用了哪种建筑风格？\"\n"
    "CoT: Let's think step by step\n"
    "\"1. 这个问题寻求的是建筑物所共有的一种建筑风格。\"\n"
    "\"2. 这些建筑是由那位在巴塞罗那建造了一座著名大教堂的建筑师设计的。\"\n"
    "\"3. 对于已知实体，只有巴塞罗那和大教堂被明确提及。\"\n"
    "\"4. 我需要按顺序发现多个未知实体。\"\n"
    "\"5. 首先，我需要确定巴塞罗那大教堂的建筑师是谁。\"\n"
    "\"6. 然后，我需要识别这位建筑师设计的其他建筑。\"\n"
    "\"7. 最后，我需要确定这些建筑物共享的建筑风格。\"\n"
    "\"8. 这些中的每一个都代表了我分析中一个不同的未知实体。\"\n"
    "So the structure is: [Core Query: What architectural style is shared Known Entities: {Subject: 大教堂, Limitation: 著名，位于巴塞罗那}, {Subject: 巴塞罗那, Limitation: 包含大教堂的城市} Unknown Entities: {Subject: 建筑师身份, Limitation: 建造了巴塞罗那大教堂}, {Subject: 其他建筑物, Limitation: 由同一位建筑师设计}, {Subject: 建筑风格, Limitation: 这些建筑物所共有的}]\n\n"
    
    "Question: \"炸药发明者出生的国家的首都是什么？\"\n"
    "CoT: Let's think step by step\n"
    "\"1.这个问题通过多步推理来询问一个首都。\"\n"
    "\"2. 唯一明确命名的实体是炸药，这是一个已知的实体。\"\n"
    "\"3. 我需要发现三个不同的信息来回答这个问题。\"\n"
    "\"4. 首先，我需要确定谁发明了炸药——这个人没有被明确命名。\"\n"
    "\"5. 然后，我需要确定这个人出生的国家。\"\n"
    "\"6. 最后，我需要确定该国家的首都。\"\n"
    "\"7. 这些中的每一个都代表了一个需要事实知识才能识别的未知实体。\"\n"
    "So the structure is: [Core Query: What is the capital Known Entities: {Subject: 炸药, Limitation: 爆炸性的发明} Unknown Entities: {Subject: 发明者身份, Limitation: 发明炸药的人}, {Subject: 出生国家, Limitation: 已确认的发明家的出生地}, {Subject: 首都, Limitation: 已确定国家的首都}]\n\n"

    "Question: \"哪个文学运动影响了那位创作了以居住在贝克街、运用演绎推理破案的角色为主角的小说的作者？\"\n"
    "CoT: Let's think step by step\n"
    "\"1. 这个问题通过一系列关联来询问一个文学运动。\"\n"
    "\"2.明确命名的实体是贝克街（一个地点）。\"\n"
    "\"3. 这个问题描述了一个具有特定特征的角色，但没有直接说出他们的名字。\"\n"
    "\"4. 它也指“那位写作的作者”——这位作者的身份并未给出。\"\n"
    "\"5. 要回答这个问题，我需要找出多条不同的信息。\"\n"
    "\"6. 首先，我需要确定哪个角色住在贝克街，并且运用推理来破解谜案。\"\n"
    "\"7. 然后，我需要确定是哪位作者创作了这个角色。\"\n"
    "\"8. 最后，我需要确定哪个文学流派影响了这位作家。\"\n"
    "\"9. 这些中的每一个都代表一个需要事实性知识的独立未知实体。\"\n"
    "So the structure is: [Core Query: What literary movement influenced Known Entities: {Subject: 贝克街, Limitation: 虚构的居住地点}, {Subject: 演绎推理, Limitation: 用于破解谜团的方法} Unknown Entities: {Subject: 角色身份, Limitation: 住在贝克街，运用演绎推理}, {Subject: 作者身份, Limitation: 创造了所识别出的角色}, {Subject: 文学运动, Limitation: 影响了已确定的作者}]\n\n"
    
    "Question: \"2010年获得最佳影片奖的那部电影的导演出生在哪个城市？\"\n"
    "CoT: Let's think step by step\n"
    "\"1. 这个问题通过一连串的关系来询问出生地。\"\n"
    "\"2. 明确命名的实体是2010年，一个时间段。\"\n"
    "\"3. 这个问题提到了“2010年获得最佳影片奖的电影”——这部电影并未被命名。\"\n"
    "\"4. 它也指“这部电影的导演”——这位导演没有名字。\"\n"
    "\"5. 我需要找出三条不同的信息来回答这个问题。\"\n"
    "\"6. 首先，我需要确定哪部电影获得了2010年的最佳影片奖。\"\n"
    "\"7. 然后，我需要确定这部电影的导演是谁。\"\n"
    "\"8. 最后，我需要确定这位导演出生在哪个城市。\"\n"
    "\"9. 这些中的每一个都代表一个需要事实性知识的独立未知实体。\"\n"
    "So the structure is: [Core Query: Which city Known Entities: {Subject: 最佳影片奖, Limitation: 2010年给出的}, {Subject: 2010, Limitation: 特定的年份} Unknown Entities: {Subject: 电影身份, Limitation:在2010年获得了最佳影片奖}, {Subject: 导演身份, Limitation: 执导了这部已确定的电影}, {Subject: 出生地, Limitation: 已确认的导演出生的城市}]\n\n"
)
    
    system_prompt = (
        "你将通过把问题分解成各个组成部分来分析问题。对于每个问题，你的回答必须严格遵循以下格式：\n\n"
        "1. 从'CoT: Let's think step by step'开始\n"
        "2. 将你的推理步骤编号为“1.”、“2.”等，每个编号都放在引号中\n"
        "3. 推理后, 写下 'So the structure is:' 随后是结构化的分解\n\n"
        
        "结构分解应包含以下三个组成部分：\n"
        "- **Core Query**: 正在寻求的主要信息。\n"
        "- **Known Entities**: 问题中明确提供的信息, 例如 {Subject: 实体, Limitation: 时间/空间/其他限制条件}.\n"
        "- **Unknown Entities**: 回答问题所需的信息，包括多跳问题中的中间步骤，其结构与已知实体的格式相同。\n\n"
        
        "核心原则：\n"
        "- 使用一致的格式： {Subject: 实体, Limitation: 限制}\n"
        "- 将两个部分中具有各自局限性的主题进行分组\n"
        "- 将时间段、地点和其他限定词作为限制条件包含在内\n"
        "- 识别所有需要的未知实体，包括多跳问题中的中间步骤\n"
        "- 区分明确提及的内容 (known) 以及隐含的/所需的 (unknown) 信息\n"
        "- 要明确哪些限制适用于哪些对象\n"
        "- 确保实体不会同时出现在已知类别和未知类别中\n\n"
        
        "您最终输出的确切格式必须是：\n"
        "CoT: Let's think step by step\n"
        "\"1. [推理步骤]\"\n"
        "\"2. [推理步骤]\"\n"
        "... 根据需要增加更多推理步骤 ...\n"
        "So the structure is: [Core Query: ... Known Entities: {Subject: 实体, Limitation: 限制}, {Subject: 其他实体, Limitation: 限制} Unknown Entities: {Subject: 实体, Limitation: 限制}, {Subject: 其他实体, Limitation: 限制}]"
    )
    
    user_message = examples_string + "\nQuestion: \"" + question + "\"\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    api_response = generate_response(messages, max_tokens=800, temperature=0)
    
    if api_response:
        # Replace multiple consecutive newlines with a single newline for easier processing
        api_response = api_response.replace("\n\n", "\n")
        
        structure_start = api_response.find("So the structure is:")
        if structure_start != -1:
            structure_text = api_response[structure_start + len("So the structure is:"):].strip()
            return structure_text
    
    return "[Core Query: Unknown Known Entities: Unknown Unknown Entities: Unknown]"




#---------------------------------------------------- Examples Database -----------------------------------------------
def get_examples_database():
    examples = [
        
        {
            "question": "法国或意大利的首都是什么？",
            "structure": "[Core Query: What is the capital Known Entities: {Subject: 法国, Limitation: 国家}, {Subject: 意大利, Limitation: 国家} Unknown Entities: {Subject: 法国首都, Limitation: 作为法国首都的城市}, {Subject: 意大利首都, Limitation: 作为意大利首都的城市}]",
            "type": "Parallel",
            "cot": "Let's think step by step\n 首先，我需要确定是否有必要进行分解。核心查询是“首都是什么”，但这个问题包含逻辑运算符“或”，连接了两个国家：法国和意大利。已知的实体是法国和意大利这两个国家。查看未知实体，我需要确定每个国家的首都。逻辑运算符“或”表明用户希望了解其中一个或两个国家的首都信息。我可以刻意忽略这些国家的其他城市或政府结构——虽然这些信息能提供有关这些国家的背景，但对确定它们的首都没有帮助。这需要进行并行分解，先独立确定每个国家的首都，然后再呈现这两种可能性的相关信息。",
            "subq1": "法国的首都是什么？",
            "subq2": "意大利的首都是什么？"
        },
        {
            "question": "瑞士和卢森堡都使用的语言是什么？",
            "structure": "[Core Query: What language is spoken in both Known Entities: {Subject: 瑞士, Limitation: 国家}, {Subject: 卢森堡, Limitation: 国家} Unknown Entities: {Subject: 瑞士语言, Limitation: 瑞士官方语言}, {Subject: 卢森堡语言, Limitation: 卢森堡官方语言}, {Subject: 共同语言, Limitation: 在两个国家都使用的语言}]",
            "type": "Parallel",
            "cot": "Let's think step by step\n 首先，我需要评估是否有必要进行分解。核心问题是“在两者都使用的语言是什么”，即寻找两个国家共同的语言。已知的实体是瑞士和卢森堡。逻辑词“两者都（BOTH）”表明我需要找到两个集合的交集。查看未知实体，我需要确定瑞士使用的语言和卢森堡使用的语言，然后找出在两个集合中都出现的语言。我可以特意忽略方言变体或语言的历史发展——尽管从语言学角度来看很有趣，但它们对确定两国共有的官方语言没有帮助。这需要进行并行分解，先分别确定每个国家的语言集合，再找出它们的交集。",
            "subq1": "瑞士使用的语言是什么？",
            "subq2": "卢森堡使用的语言是什么？"
        },
        {
        "question": "孛剌合真（窝阔台的妻子）的公公是谁？",
        "structure": "[Core Query: Who is person's father-in-law Known Entities: {Subject: 孛剌合真, Limitation: 窝阔台的妻子}, {Subject: 窝阔台, Limitation: 孛剌合真的丈夫} Unknown Entities: {Subject: 岳父身份, Limitation: 窝阔台的父亲, 孛剌合真的配偶的父亲}, {Subject: 家庭关系链, Limitation: 配偶关系将孛剌合真与窝阔台的父亲联系起来}]",
        "type": "Sequential",
        "cot": "Let's think step by step\n 首先，我需要确定是否有必要进行分解。核心问题是“某人的岳父是谁”。已知的实体是“孛剌合真”和“窝阔台”。查看未知实体，我需要确定孛剌合真的岳父是谁。由于孛剌合真被明确认定为窝阔台的妻子，她的岳父应该是窝阔台的父亲。我可以刻意忽略关于这些人物的历史背景——虽然这可能会提供有趣的背景信息，但对确定窝阔台的父亲是谁并无帮助。这需要进行顺序分解，因为要找到孛剌合真的岳父，首先得确定孛剌合真的配偶是谁，然后再确定该配偶的父亲是谁。",
        "subq1": "博剌忽真的配偶是谁？",
        "subq2": "谁是[answer_subquestion1]的父亲？"
        },

        {
            "question": "《第九交响曲》（《合唱交响曲》）的作曲家是在《星月夜》的画家之后去世的吗？",
            "structure": "[Core Query: Did person A die after person B Known Entities: {Subject: 交响乐, Limitation: \"第九交响曲\" （《合唱交响曲》）}, {Subject: 绘画作品, Limitation: \"《星月夜》\"} Unknown Entities: {Subject: 作曲家身份, Limitation: 第九交响曲的}, {Subject: 画家身份, Limitation: 《星月夜》的}, {Subject: 死亡日期, Limitation: 已确认的作曲家的}, {Subject: 死亡日期, Limitation: 已确认画家的}, {Subject: 出生日期, Limitation: 两位艺术家的}, {Subject: 艺术风格, Limitation:两位艺术家的}]",
            "type": "Parallel",
            "cot": "Let's think step by step\n 首先，我需要确定是否有必要进行分解。核心查询是比较两个死亡日期：“A人物是否在B人物之后去世”。已知实体是作品《第九交响曲》和《星月夜》。查看未知实体，我需要作曲家身份、画家身份以及他们的死亡日期来进行比较。我可以刻意忽略出生日期和艺术风格——虽然这些信息可能对了解背景有帮助，但对确定谁先去世谁后去世并无帮助。这需要进行并行分解，因为我需要先分别找到每个人的死亡日期，然后再进行比较。",
            "subq1": "《第九交响曲》（《合唱交响曲》）的作曲家是何时去世的？",
            "subq2": "《星月夜》的画家是何时去世的？"
        },
        
        {
            "question": "《第九交响曲》（《合唱交响曲》）的作曲家小时候演奏的是哪种乐器？",
            "structure": "[Core Query: Which instrument did person play as a child Known Entities: {Subject: 交响乐, Limitation: \"第九交响曲\" （《合唱交响曲》）} Unknown Entities: {Subject: 作曲家身份, Limitation: 第九交响曲的}, {Subject: 乐器, Limitation: 作曲家小时候演奏的}, {Subject: 音乐教育, Limitation: 作曲家的音乐训练}, {Subject: 作曲风格, Limitation: 作曲家作品的特点}]",
            "type": "Sequential",
            "cot": "Let's think step by step\n 首先，我需要评估是否有必要进行分解。核心问题是“这个人小时候演奏什么乐器”。已知的实体是《第九交响曲》。查看未知实体，我需要先确定作曲家是谁，然后再确定他们小时候演奏什么乐器。我可以刻意忽略音乐教育和作曲风格——尽管它们与作曲家的成长相关，但并不直接回答演奏的具体乐器是什么。这需要按顺序进行分解，因为要确定小时候演奏的乐器，首先得知道作曲家是谁。",
            "subq1": "谁创作了《第九交响曲》（《合唱交响曲》）？",
            "subq2": "[answer_subquestion1]小时候演奏的是哪种乐器？"
        },
        
        {
            "question": "《1984》这部反乌托邦小说的作者是在哪一年去世的？",
            "structure": "[Core Query: In which year did person pass away Known Entities: {Subject: 小说, Limitation: 反乌托邦} Unknown Entities: {Subject: 作者身份, Limitation: 小说《1984》的}, {Subject: 逝世年份, Limitation: 已确认作者的}, {Subject: 其他作品, Limitation: 由同一作者所著}, {Subject: 政治观点, Limitation: 影响小说的作者}]",
            "type": "Sequential",
            "cot": "Let's think step by step\n 首先，我需要评估是否有必要进行分解。核心查询是“某人在哪一年去世”。已知实体是小说《1984》。查看未知实体，我需要先确定作者是谁，然后确定他们的死亡年份。我可以刻意忽略其他作品和政治观点——虽然这些提供了关于作者职业生涯和影响的信息，但它们并不直接帮助确定作者的死亡年份。这需要按顺序进行分解，因为找到死亡年份依赖于首先识别出作者。",
            "subq1": "谁创作了反乌托邦小说《1984》？",
            "subq2": "在[answer_subquestion1]去世的那一年是哪一年？"
        },
        
        {
            "question": "库拉姆加希和特罗伊克里斯蒂都位于同一个国家吗？",
            "structure": "[Core Query: Are both located in the same country Known Entities: {Subject: 库拉姆加希, Limitation: 地点名称}, {Subject: 特罗伊克里斯蒂, Limitation: 地点名称} Unknown Entities: {Subject: 城市, Limitation: 包含库拉姆加希}, {Subject: 城市, Limitation: 包含特罗伊克里斯蒂}, {Subject:地理特征, Limitation: 两个地点的}, {Subject: 人口数据, Limitation: 两个地点的}]",
            "type": "Parallel",
            "cot": "Let's think step by step\n 首先，我需要评估是否有必要进行分解。核心问题是“两个地点是否位于同一个国家”，这是一个比较问题。已知实体是地点“库拉姆加希”和“特罗伊克里斯蒂”。查看未知实体，我需要确定每个地点所属的国家以进行比较。我可以刻意忽略地理特征和人口数据——虽然这些提供了关于地点的背景信息，但它们并不帮助确定这些地点属于哪个国家。这需要并行分解，以独立地确定每个地点所属的国家后再进行比较。",
            "subq1": "库拉姆加希位于哪个国家？",
            "subq2": "特罗伊克里斯蒂位于哪个国家？"
        },
        
        {
            "question": "1995年电影《Coolie No. 1》的导演和电影《The Sensational Trial》的导演国籍相同吗？",
            "structure": "[Core Query: Do person A and person B have the same nationality Known Entities: {Subject: 电影A, Limitation: 《Coolie No. 1》（1995年）}, {Subject: 电影B, Limitation: 《The Sensational Trial》} Unknown Entities: {Subject: A导演的身份, Limitation: 《Coolie No. 1》的}, {Subject: B导演的身份, Limitation: 《The Sensational Trial》的}, {Subject: 国籍, Limitation: A导演的}, {Subject: 国籍, Limitation: B导演的}, {Subject: 电影类型, Limitation: 两部电影的}, {Subject: 票房表现, Limitation: 两部电影的}]",
            "type": "Parallel",
            "cot": "Let's think step by step\n 首先，我需要确定是否有必要进行分解。核心问题是询问两个人是否拥有相同的国籍。已知实体是电影《Coolie No. 1》和《The Sensational Trial》。查看未知实体，我需要确定每位导演及其国籍，以便进行比较。我可以特意忽略电影类型和票房表现——虽然这些能提供有关电影的背景信息，但与导演的国籍无关。这需要进行并行分解，因为我可以先独立找出每位导演的国籍，然后再对他们进行比较。",
            "subq1": "1995年电影《Coolie No. 1》的导演的国籍是什么？",
            "subq2": "电影《The Sensational Trial》的导演的国籍是什么？"
        },
        
        {
            "question": "巴西的官方货币是什么？",
            "structure": "[Core Query: What is the official currency Known Entities: {Subject: 巴西, Limitation: 城市} Unknown Entities: {Subject: 货币, Limitation: 巴西官方}, {Subject: 货币史, Limitation: 巴西以前的货币}, {Subject: 汇率, Limitation: 相对于美元的价值}, {Subject: 经济指标, Limitation: 通货膨胀，国内生产总值}]",
            "type": "None",
            "cot": "Let's think step by step\n 首先，我需要评估是否真的有必要进行分解。核心问题是“官方货币是什么”。已知的实体是“巴西”。查看未知实体，我只需要确定巴西当前的官方货币。我可以刻意忽略货币历史、汇率和经济指标——虽然它们能提供有关巴西经济的背景信息，但对确定当前的官方货币没有帮助。这是一个简单的事实性问题，一步就能回答，无需分解。",
            "subq1": "巴西的官方货币是什么？",
            "subq2": ""
        },
        
        {
            "question": "哪位作曲家的歌剧首演年份与法国大革命爆发年份相同？这位作曲家死于哪个世纪？",
            "structure": "[Core Query: In which century did person die Known Entities: {Subject: 法国大革命, Limitation: 有具体起始年份的历史事件} Unknown Entities: {Subject: 作曲家身份, Limitation: 谁的歌剧首演年份与法国大革命开始的年份相同}, {Subject: 死亡世纪, Limitation: 已确认的作曲家的}, {Subject: 歌剧详情, Limitation: 标题和音乐风格}, {Subject: 作曲家的国籍, Limitation: 原产国}]",
            "type": "Sequential",
            "cot": "Let's think step by step\n 首先，我需要评估是否有必要进行分解。核心问题是“这个人死于哪个世纪”。已知的实体是“法国大革命”。查看未知实体，我需要确定哪位作曲家的歌剧首演恰逢法国大革命开始，然后确定他们死于哪个世纪。我可以刻意忽略歌剧的细节和作曲家的国籍——虽然这些信息提供了关于作曲家的背景，但对确定他们的死亡时间没有帮助。这需要按顺序进行分解，因为要确定死亡的世纪，首先得确定具体的作曲家。",
            "subq1": "哪位作曲家的歌剧首演年份与法国大革命开始的年份相同？",
            "subq2": "[answer_subquestion1]死于哪个世纪？"
        },
        
        {
            "question": "这座地标性建筑由设计了著名玻璃金字塔的建筑师设计，它是在哪一年完工的？",
            "structure": "[Core Query: In what year was building completed Known Entities: {Subject:玻璃金字塔, Limitation: 著名的建筑作品} Unknown Entities: {Subject: 建筑师身份, Limitation: 建造了玻璃金字塔}, {Subject:标志性建筑, Limitation: 由知名建筑师设计}, {Subject: 完成年份, Limitation: 已识别的标志性建筑的}, {Subject: 建筑风格, Limitation: 两种结构的}, {Subject: 建筑材料, Limitation: 在两种结构中都有使用}]",
            "type": "Sequential",
            "cot": "Let's think step by step\n 首先，我需要确定是否有必要进行分解。核心查询是“这座建筑是哪一年完工的”。已知实体是“玻璃金字塔”。查看未知实体，我需要确定设计玻璃金字塔的建筑师，然后找到他们的标志性建筑，最后确定该建筑的完工年份。我可以刻意忽略建筑风格和建筑材料——虽然这些能提供有关建筑的有趣背景信息，但对确定完工年份没有帮助。由于确定建筑师、建筑以及其完工年份之间存在依赖关系，这需要进行顺序分解。",
            "subq1": "哪位建筑师设计了著名的玻璃金字塔？",
            "subq2": "由[answer_subquestion1]设计的这座标志性建筑是在哪一年完工的？"
        },
        
        {
            "question": "谁是第一个进入太空的女性宇航员？",
            "structure": "[Core Query: Who was the first female astronaut Known Entities: {Subject: 太空旅行, Limitation: 由女性完成} Unknown Entities: {Subject: 宇航员身份, Limitation: 女性，首位进入太空的人}, {Subject: 发布日期, Limitation:首次女性太空任务的}, {Subject: 航天器, Limitation: 用于首次女性太空任务}, {Subject: 任务持续时间, Limitation: 首次女性太空飞行的时长}]",
            "type": "None",
            "cot": "Let's think step by step\n 首先，我需要评估是否真的有必要进行分解。核心问题是“谁是第一位女宇航员”。已知的实体是“太空旅行”。查看未知实体，我只需要确定哪位女性是第一位进入太空的人。我可以特意忽略发射日期、航天器和任务持续时间——虽然这些信息提供了关于这一历史性任务的有趣背景，但它们对确定谁是第一位女宇航员没有帮助。这是一个简单的事实性问题，无需分解，一步就能回答。",
            "subq1": "谁是第一个进入太空的女性宇航员？",
            "subq2": ""
        },
        
        {
            "question": "当特斯拉的CEO也成为此前名为Twitter的社交媒体平台的所有者时，那是在什么时候？",
            "structure": "[Core Query: When became owner Known Entities: {Subject: 特斯拉CEO, Limitation: 具有特定角色的人}, {Subject: 社交媒体平台, Limitation: 以前称为推特} Unknown Entities: {Subject: 收购日期, Limitation: 当CEO成为平台所有者时}, {Subject: 购买价格, Limitation: 收购支付金额}, {Subject: 平台变更, Limitation: 收购后的修改}]",
            "type": "None",
            "cot": "Let's think step by step\n 首先，我需要评估是否真的有必要进行分解。核心问题是“当特斯拉CEO成为Twitter平台的所有者时，那是在什么时候？”。已知实体是“特斯拉CEO”和“此前名为Twitter的社交媒体平台”。查看未知实体，我只需要确定收购日期以回答所有权变更的时间。我可以刻意忽略购买价格和平台变更——虽然这些信息提供了关于收购及其后续影响的背景，但它们对确定所有权转移时间没有帮助。这是一个简单的事实性问题，无需分解，一步就能回答。",
            "subq1": "当特斯拉的CEO也成为此前名为Twitter的社交媒体平台的所有者时，那是在什么时候？",
            "subq2": ""
        },
        
    ]
    return examples


#---------------------------------------------------- Vector Similarity -----------------------------------------------
def find_similar_examples(question, examples, num_examples=3):
    example_questions = [ex["question"] for ex in examples]
    vectorizer = TfidfVectorizer()
    all_questions = example_questions + [question]
    vectors = vectorizer.fit_transform(all_questions)
    question_vector = vectors[-1]
    example_vectors = vectors[:-1]
    similarities = cosine_similarity(question_vector, example_vectors)[0]
    most_similar_indices = similarities.argsort()[-num_examples:][::-1]
    return [examples[i] for i in most_similar_indices]

#---------------------------------------------------- Prompt Construction -----------------------------------------------
def construct_prompt(question, examples, structure):
    prompt = """我希望你能对问题进行分析，并将它们分解成子问题。
我会提供问题及其结构分析。 (Core Query, Known Entities, Limiting Conditions, and Unknown Entities).
请分析问题结构并确定如何对其进行分解。请遵循以下格式：

    Question: [The original question]

Structure: [Analysis of core query, known entities, limiting conditions, and unknown entities]

CoT: Let's think step by step
[对问题结构进行详细的分步推理分析]
[分析核心查询、已知实体和限制条件]
[评估哪些未知实体对于回答该问题至关重要]
[判断是否需要分解以及应采用何种策略]
[确保子问题中保留关键的限制条件]

So the Type is: [Parallel, Sequential, or None]

So the Subquestion 1 is: [First subquestion; if type is None, should be identical to original question]

So the Subquestion 2 is: [Second subquestion; if Sequential, MUST include [answer_subquestion1]; if Parallel, MUST NOT reference subquestion 1; if None, leave empty]

**强制性要求：**

1. **DECOMPOSITION RULE**: 仅在必要时将其分解为两个子问题.

2. **SIMPLICITY EVALUATION**: 一步即可回答的问题应归类为 "None" 类型.

3. **SUBQUESTION CLARITY**: 每个子问题都必须得出一个具体、真实且明确的答案。

4. **NONE TYPE FORMATTING**: Subquestion 1 必须与原始问题匹配， Subquestion 2必须为空。

5. **SEQUENTIAL REQUIREMENTS**: Subquestion 2 必须包含 [answer_subquestion1] 占位符并形成一个完整的逻辑链。

6. **PARALLEL REQUIREMENTS**: 这两个子问题必须相互独立，且合起来能解决原问题。

7. **PRECISION**: 子问题必须包含足够的上下文，以便无需澄清就能得到解答。

8. **TERMINOLOGY CONSISTENCY**: 始终统一使用 "subquestion"。

9. **ENTITY FOCUS**: 只包含直接有助于回答核心查询的实体。

10. **CONSISTENCY**: 你的推理必须与你最终的分解一致。

11. **MEANINGFUL DECOMPOSITION**: 避免无关紧要的定义性子问题。

12. **SUBSTANTIVE CONTRIBUTION**: 每个子问题都必须提供有助于推进解决方案的信息。

13. **PRESERVING JOINT LIMITING CONDITIONS**: 当多个限制条件共同定义一个单一实体时，这些条件必须放在一起，绝不能拆分到子问题中。像“两者都”“相同”“一起”“共同”以及类似的词汇表明，限制性条件应作为一个整体保留。 像“谁既发明了X又发明了Y”或“谁因A和B而闻名”这类问题，不应该被分解成关于独立实体的平行问题。
14. **DECOMPOSITION VALIDATION**:在最终确定之前，务必验证你的分解的正确性和完整性。

以下是一些例子：
"""
    for ex in examples:
        prompt += f"""
    Question: {ex["question"]}

Structure: {ex["structure"]}

CoT: {ex["cot"]}

So the Type is: {ex["type"]}

So the Subquestion 1 is: {ex["subq1"]}"""
        if ex["type"] != "None":
            prompt += f"\nSo the Subquestion 2 is: {ex['subq2']}"
        prompt += "\n"
    prompt += f"""
### **现在，分析下面这个问题：**
    Question: {question}

Structure: {structure}

CoT: Let's think step by step
"""
    return prompt

#---------------------------------------------------- API Client -----------------------------------------------
def generate_responses(prompt, api_url=None, max_tokens=800, temperature=0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    """Using OpenAI client instead of original API call"""
    system_message = """你是一位擅长分析问题并将其分解为更简单子问题的专家。
你仔细区分了顺序性问题（其中第二个子问题取决于第一个子问题的答案），
并行问题（其中两个子问题都可以独立回答），以及 None-type 问题（这些问题已经很简单了）。

**您必须执行的关键要求：**

1. **DECOMPOSITION NECESSITY**: 只有在绝对必要时才拆解问题。如果一个问题可以直接回答，就将其标记为 "None" 类型。

2. **SEQUENTIAL DECOMPOSITION CORRECTNESS**: 对于序列类型，验证将 Subquestion 1 的答案直接代入 Subquestion 2 是否能得出原始问题的完整答案。这是对有效顺序分解的最重要测试。

3. **NO TRIVIAL SUBQUESTIONS**: 绝不要创建诸如“X是谁/是什么？”这类基础性的定义问题。 除非对于中间推理来说绝对必要，并且不属于常识。

4. **LOGICAL PATHWAY**: 确保子问题与原始问题之间存在清晰、直接的逻辑联系。每一步都必须朝着最终答案推进。

5. **ALIGNMENT BETWEEN COT AND DECOMPOSITION**: 你的 step-by-step 推理 必须与你最终的分解选择和子问题表述完全一致。

6. **VALIDATION CHECK**: 在最终确定之前，务必在脑海中代入预期答案，验证你的分解结构是否有效。

7. **FOCUS ON DIRECT ANSWERABLE QUESTIONS**: 每个子问题都必须产生一个具体、真实的答案，直接有助于解决原始问题。

8. **PRESERVING JOINT LIMITING CONDITIONS**: 绝不能拆分共同定义一个单一实体的多个限制条件，这一点至关重要。 当问题中出现“X和Y两者”“相同”“一起”“共同”或其他类似表示单一实体具有多种属性的术语时，请勿将其拆分为单独的问题。例如：

   - “谁发明了X和Y？”不应该拆分成“谁发明了X？”和“谁发明了Y？”
   - “哪个国家同时具有特征A和特征B？”必须作为一个单独的问题保留
   - “‘什么以同时做X和Y而闻名？’必须保持原样”

   此要求优先于其他分解考量。当多个条件共同定义我们要寻找的内容时，在任何子问题中都必须将它们一并保留。未能维持这些联合条件会使分解完全失效。
""".strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    return generate_response(messages, max_tokens, temperature, top_p, frequency_penalty, presence_penalty)

#---------------------------------------------------- Answer Function -----------------------------------------------


#---------------------------------------------------- Get Final Answer Function -----------------------------------------------
def construct_final_prompt(current_question: str, current_sub_questions: List[Tuple[str, str]]) -> str:
    # Function to construct the final prompt remains unchanged
    examples = [
        {
            "question": "Which mountain is taller, Mount Everest or K2?",
            "sub_questions": [
                ("What is the height of Mount Everest?", "8,848 meters"),
                ("What is the height of K2?", "8,611 meters")
            ],
            "cot": "Mount Everest has a height of 8,848 meters, and K2 has a height of 8,611 meters. Since 8,848 meters is greater than 8,611 meters, Mount Everest is taller.",
            "final_answer": "Mount Everest"
        },
        {
            "question": "Who invented the telephone?",
            "sub_questions": [
                ("Which invention is known as the telephone?", "A device for voice communication"),
                ("Who is credited with creating the telephone?", "Alexander Graham Bell")
            ],
            "cot": "The telephone is a device for voice communication, and Alexander Graham Bell is credited with creating it.",
            "final_answer": "Alexander Graham Bell"
        },
          {
            "question": "What is the capital city of the country that hosts the annual Tour de France?",
            "sub_questions": [
                ("Which country hosts the annual Tour de France?", "France"),
                ("What is the capital city of France?", "Paris")
            ],
            "cot": "The country that hosts the annual Tour de France is France, and the capital city of France is Paris.",
            "final_answer": "Paris"
        },
        {
    "question": "How many players are on a standard soccer team during a match?",
    "sub_questions": [
        ("What is a standard soccer team?", "A team in an official soccer match"),
        ("How many players are on a standard soccer team?", "11")
    ],
    "cot": "A standard soccer team is a team in an official soccer match, and it has 11 players. Since the question asks 'how many players', the answer should include the unit 'players'.",
    "final_answer": "11 players"
},
        {
            "question": "Is the Pacific Ocean larger than the Atlantic Ocean?",
            "sub_questions": [
                ("What is the size of the Pacific Ocean?", "155.6 million square kilometers"),
                ("What is the size of the Atlantic Ocean?", "106.5 million square kilometers")
            ],
            "cot": "The Pacific Ocean has a size of 155.6 million square kilometers, and the Atlantic Ocean has a size of 106.5 million square kilometers. Since 155.6 million square kilometers is greater than 106.5 million square kilometers, the Pacific Ocean is larger.",
            "final_answer": "yes"
        },
        {
            "question": "When did the fictional character James Barker win his first Nobel Prize?",
            "sub_questions": [
                ("Who is James Barker?", "No reliable information found about a notable person named James Barker winning a Nobel Prize"),
                ("When did James Barker win his first Nobel Prize?", "No information available")
            ],
            "cot": "Based on the subquestions, there is no reliable information about a notable person named James Barker winning a Nobel Prize. The question appears to be asking about a fictional character or contains incorrect assumptions. Without concrete information about this person and their achievements, I cannot determine when they won their first Nobel Prize, or if they won one at all.",
            "final_answer": "[none]"
        },
        {
            "question": "Who was the first person to reach the summit of Mount Everest or K2?",
            "sub_questions": [
                ("Who was the first person to reach the summit of Mount Everest?", "Edmund Hillary and Tenzing Norgay in 1953"),
                ("Who was the first person to reach the summit of K2?", "[none]")
            ],
            "cot": "For Mount Everest, Edmund Hillary and Tenzing Norgay were the first to reach the summit in 1953. For K2, no information is available. Since the question asks for the first person to reach EITHER Mount Everest OR K2, and we have valid information for Mount Everest but not for K2, we can provide the answer based on the available information about Mount Everest.",
            "final_answer": "Edmund Hillary and Tenzing Norgay in 1953"
        }
    ]

    prompt = """You are an expert at answering questions based on provided subquestions and their answers. 

IMPORTANT GUIDELINES:
1. If you cannot extract the necessary information to answer the question, please respond with '[none]'.
2. Your answer MUST follow the structure "so the Final answer is: [your answer]".
3. First provide your chain of thought reasoning in the CoT section, then give your final answer.
Below are some examples to guide you:\n\n"""
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Question: {example['question']}\n"
        prompt += "Subquestions:\n"
        for sub_q, sub_a in example['sub_questions']:
            prompt += f"{sub_q}: {sub_a}\n"
        prompt += f"CoT: {example['cot']}\n"
        prompt += f"so the Final answer is: {example['final_answer']}\n\n"

    prompt += "Now, based on the following question and subquestions, generate the final answer:\n\n"
    prompt += f"Question: {current_question}\n"
    prompt += "Subquestions:\n"
    for sub_q, sub_a in current_sub_questions:
        prompt += f"{sub_q}: {sub_a}\n" 
    prompt += """
CoT: 
"""
    return prompt

def get_final_answer(question: str, sub_questions: List[Tuple[str, str]], api_url=None) -> str:
    """Using OpenAI client to get the final answer"""
    prompt = construct_final_prompt(question, sub_questions)
    
    messages = [
        {"role": "system", "content": "You are an expert at answering questions based on provided subquestions and their answers."},
        {"role": "user", "content": prompt}
    ]
    
    return generate_response(messages, max_tokens=800, temperature=0)

#---------------------------------------------------- Parsing Response -----------------------------------------------
def parse_decomposition_response(response):
    # Function to parse decomposition response remains unchanged
    type_match = re.search(r'So the Type is:\s*(\w+)', response, re.IGNORECASE)
    question_type = type_match.group(1) if type_match else "None"
    
    # Extract Subquestion 1, ensuring only a single line of question content
    subq1_match = re.search(r'So the Subquestion 1 is:\s*(.+)', response, re.DOTALL)
    subq1 = subq1_match.group(1).strip() if subq1_match else ""
    # Clean up possible next line content
    subq1 = subq1.split('\n')[0].strip()  # Take only the first line
    
    subq2 = ""
    if question_type.lower() != "none":
        # Extract Subquestion 2, ensuring only a single line of question content
        subq2_match = re.search(r'So the Subquestion 2 is:\s*(.+)', response, re.DOTALL)
        subq2 = subq2_match.group(1).strip() if subq2_match else ""
        subq2 = subq2.split('\n')[0].strip()  # Take only the first line
        if not subq2 and question_type.lower() != "none":
            question_type = "None"
            
        # Check if subq2 contains [answer_subquestion1], if so ensure type is Sequential
        if "[answer_subquestion1]" in subq2:
            if question_type.lower() != "sequential":
                print(f"Detected subquestion2 containing reference to subquestion1 answer, correcting type from {question_type} to Sequential.")
                question_type = "Sequential"
        # If type is Sequential but subq2 doesn't reference subq1's answer, issue a warning but don't modify the type
        elif question_type.lower() == "sequential" and "[answer_subquestion1]" not in subq2:
            print(f"Warning: Type is Sequential but subquestion2 doesn't reference subquestion1's answer.")
    
    return {"type": question_type, "subq1": subq1, "subq2": subq2}

#---------------------------------------------------- Binary Tree Structure -----------------------------------------------
global_node_counter = 0
class QuestionNode:
    def __init__(self, question, q_type="None", subq1=None, subq2=None, parent=None, is_left_child=True, question_id=None):
        global global_node_counter
        # 如果没有提供ID，生成一个唯一ID
        if question_id is None:
            self.id = f"N{global_node_counter}"
            global_node_counter += 1
        else:
            self.id = question_id
        
        self.question = question
        self.display_question = question
        self.type = q_type
        self.left = None
        self.right = None
        self.parent = parent
        self.is_left_child = is_left_child
        self.depends_on = None
        self.subq1_text = subq1
        self.subq2_text = subq2
        self.answer = None  # Store the node's answer

    def __str__(self):
        return f"ID: {self.id}, Type: {self.type}, Q: {self.display_question[:50]}{'...' if len(self.display_question) > 50 else ''}"

#---------------------------------------------------- Extract Answer from Response -----------------------------------------------
def extract_answer(full_response):
    match = re.search(r'so the answer is:\s*(.*)', full_response, re.IGNORECASE)
    if match:
        raw_answer = match.group(1).strip()
        # Clean up the answer, remove extra symbols
        cleaned_answer = re.sub(r'["*]+$', '', raw_answer)
        # Handle [none] special case
        if cleaned_answer.lower() == '[none]':
            return '[none]'
        return cleaned_answer
    return "Answer not found"

#---------------------------------------------------- Tree Building and Traversal -----------------------------------------------
def build_question_tree(question, api_url=None, max_tokens=800, temperature=0.2, top_p=1.0, frequency_penalty=0.0, 
                        presence_penalty=0.0, examples_db=None, num_examples=3, depth=0, 
                        parent=None, is_left_child=True, max_height=3, placeholder_answers=None):
    """
    Build a question tree through recursive decomposition
    
    Parameters:
    - question: The question to decompose
    - api_url, max_tokens, temperature, top_p, frequency_penalty, presence_penalty: API parameters
    - examples_db: Database of examples for finding similar examples
    - num_examples: Number of similar examples to use
    - depth: Current depth in the tree (0 for root)
    - parent: Parent node
    - is_left_child: Whether the current node is a left child
    - max_height: Maximum allowed height of the tree (1 = root only)
    - placeholder_answers: Dictionary of previously computed answers
    
    Returns:
    - node: The root node of the built tree
    """
    global global_node_counter
    
 
    
    # Initialize placeholder_answers dictionary for storing replacement values
    if placeholder_answers is None:
        placeholder_answers = {}
    
    # Check if we've exceeded the maximum height
    # depth starts at 0 for root, so if max_height is 3, we allow depths 0, 1, and 2
    if depth >= max_height:
        
        # Create a leaf node with a unique ID using the global counter
        leaf_node = QuestionNode(
            question=question, 
            q_type="None",
            subq1=question, 
            subq2="",
            parent=parent, 
            is_left_child=is_left_child
        )
        
        
        
        # Handle replacement markers
        if parent and "[answer_subquestion1]" in leaf_node.question:
            if parent.type == "Sequential" and not is_left_child:
                leaf_node.depends_on = parent.left.id if parent.left else None
                leaf_node.display_question = leaf_node.question.replace("[answer_subquestion1]", f"[answer from {leaf_node.depends_on}]")

        elif parent and "[answer from" in leaf_node.question: 
            match = re.search(r'\[answer from (.*?)\]', leaf_node.question)
            if match:
                leaf_node.depends_on = match.group(1)
                
        
   
        return leaf_node
    
    # Handle replacement markers, replace if replacement values are available
    modified_question = question
    depends_on = None
    
    if "[answer_subquestion1]" in question:
       
        if parent and parent.type == "Sequential" and not is_left_child and parent.left:
            depends_on = parent.left.id
           
            # Check if there's a replacement value available
            if depends_on in placeholder_answers:
                modified_question = question.replace("[answer_subquestion1]", placeholder_answers[depends_on])
                
    elif "[answer from" in question:
        
        match = re.search(r'\[answer from (.*?)\]', question)
        if match:
            depends_on = match.group(1)
           
            # Check if there's a replacement value available
            if depends_on in placeholder_answers:
                modified_question = question.replace(f"[answer from {depends_on}]", placeholder_answers[depends_on])
               
    
    # If the question still contains replacement markers and there's no replacement value available, create a node to be replaced
    if ("[answer_subquestion1]" in modified_question or "[answer from" in modified_question) and not "[answer from" in modified_question:
       
        node = QuestionNode(
            question=question, 
            q_type="None",
            subq1=question, 
            subq2="",
            parent=parent, 
            is_left_child=is_left_child
        )
        
        if depends_on:
            node.depends_on = depends_on
          
            if "[answer_subquestion1]" in node.question:
                node.display_question = node.question.replace("[answer_subquestion1]", f"[answer from {node.depends_on}]")
               

        return node
    
    # If examples_db is None, get the examples database
    if examples_db is None:
        examples_db = get_examples_database()
      
    
    # Analyze question structure and build decomposition prompt
  
    structure = analyze_question_structure(modified_question)
    print("++++++++++++++++++++++++")
    print(f"Question structure analysis (depth={depth}):", structure)
    similar_examples = find_similar_examples(modified_question, examples_db, num_examples)

    prompt = construct_prompt(modified_question, similar_examples, structure)

    response = generate_responses(prompt, api_url, max_tokens, temperature, top_p, frequency_penalty, presence_penalty)
    print(f"Decomposition response (depth={depth}):", response)

    decomposition = parse_decomposition_response(response)
  

    # Create a new node with a unique ID
    node = QuestionNode(
        question=question, 
        q_type=decomposition['type'],
        subq1=decomposition['subq1'], 
        subq2=decomposition['subq2'],
        parent=parent, 
        is_left_child=is_left_child
    )
    
   
    if parent:
        print(f"Debugging - parent Node {parent.id}")
    
    # Use the modified question as the display question
    if modified_question != question:
        node.display_question = modified_question
        
    
    # Handle replacement markers
    if depends_on:
        node.depends_on = depends_on
       

    # Check if we need to stop expansion at this level to respect max_height
    # If we're at depth max_height-1, the next level would be at max_height
    # So we force this node to be a leaf by setting its type to "None"
    if depth >= max_height - 1:
        # Force node type to be "None" to prevent further decomposition
        node.type = "None"
        
        return node

    if node.type != "None":
      
        # First handle the left subtree
       
        left_node = build_question_tree(
            node.subq1_text, api_url, max_tokens, temperature, top_p,
            frequency_penalty, presence_penalty, examples_db, num_examples,
            depth + 1, node, True, max_height, placeholder_answers
        )
        node.left = left_node
        
        if node.left:
            print(f"Debugging - left Child node id: {node.left.id}, type: {node.left.type}")
        
        # Handle the right subtree
        if node.type == "Sequential":
            
            # For sequential type, check if the right subquestion contains replacement markers
            right_question = node.subq2_text
            
            right_node = build_question_tree(
                right_question, api_url, max_tokens, temperature, top_p,
                frequency_penalty, presence_penalty, examples_db, num_examples,
                depth + 1, node, False, max_height, placeholder_answers
            )
            node.right = right_node
            
            if node.right:
                print(f"Debugging - right Child node id: {node.right.id}, type: {node.right.type}")
            
            # Set up dependency relationships for sequential type
            if node.right and node.left:
                
                if node.right.depends_on is None and "[answer_subquestion1]" in node.right.question:
                    node.right.depends_on = node.left.id
                    node.right.display_question = node.right.question.replace("[answer_subquestion1]", f"[answer from {node.right.depends_on}]")
                    
        else:  # Parallel type
            
            right_node = build_question_tree(
                node.subq2_text, api_url, max_tokens, temperature, top_p,
                frequency_penalty, presence_penalty, examples_db, num_examples,
                depth + 1, node, False, max_height, placeholder_answers
            )
            node.right = right_node
            
            if node.right:
                print(f"Debugging - right Child node id: {node.right.id}, type: {node.right.type}")

    # Simple tree height and node count estimation
    has_children = node.left is not None or node.right is not None
    
    if has_children:
        print(f"Debugging - left child node: {' Exists' if node.left else 'does not exist '}, right child node: {' Exists' if node.right else' does not exist '}")
    
    
    return node



def get_all_nodes_postorder(node):
    """Get all nodes in post-order traversal"""
    if node is None:
        return []
    left_nodes = get_all_nodes_postorder(node.left)
    right_nodes = get_all_nodes_postorder(node.right)
    return left_nodes + right_nodes + [node]

def print_all_nodes(node, indent=""):
    # Function to print all nodes remains unchanged
    print(f"{indent}Node ID: {node.id}")
    print(f"{indent}Type: {node.type}")
    print(f"{indent}Question: {node.display_question}")
    if node.depends_on:
        print(f"{indent}Depends on: {node.depends_on}")
    if node.answer:
        print(f"{indent}Answer: {node.answer}")
    if node.left:
        print(f"{indent}Left child:")
        print_all_nodes(node.left, indent + "  ")
    if node.right:
        print(f"{indent}Right child:")
        print_all_nodes(node.right, indent + "  ")

#---------------------------------------------------- Tree Solving -----------------------------------------------
def solve_tree(root, original_question, api_url=None, max_tokens=4000, 
              temperature=0, top_p=1.0, frequency_penalty=0.0, 
              presence_penalty=0.0, examples_db=None, num_examples=20,
              enhanced_right_subtree=True, right_subtree_variants=2, 
              right_subtree_trees_per_variant=2, max_height=3,
              placeholder_answers=None):
    """
    Solve a single problem tree and return the answer, supporting the construction of an enhanced right subtree
    """
    global global_node_counter
    
    if placeholder_answers is None:
        placeholder_answers = {}  
    
    
    processed_node_ids = set()
    
    
    def solve_node(node, updated_tree=False, current_depth=0):
        if node is None:
            return {}

        
        if node.id in processed_node_ids:
            return {node.id: placeholder_answers.get(node.id, "[none]")}
        
        
        processed_node_ids.add(node.id)
        
        node_answers = {}
        
        
        if node.depends_on and node.depends_on not in placeholder_answers:
            
            def find_node_by_id(search_node, target_id):
                if search_node is None:
                    return None
                if search_node.id == target_id:
                    return search_node
                left_result = find_node_by_id(search_node.left, target_id)
                if left_result:
                    return left_result
                return find_node_by_id(search_node.right, target_id)
            
            dependent_node = find_node_by_id(root, node.depends_on)
            if dependent_node:
                dependent_answers = solve_node(dependent_node, updated_tree, current_depth)
                node_answers.update(dependent_answers)
        
       
        if node.id in placeholder_answers:
            node.answer = placeholder_answers[node.id]
            node_answers[node.id] = node.answer
            
            return node_answers
        
      
        if (node.left is None and node.right is None) or node.type == "None":
            actual_question = node.question
            
           
            if node.depends_on and node.depends_on in placeholder_answers:
                dependent_answer = placeholder_answers[node.depends_on]
                
                if dependent_answer.lower() == "[none]":
                    node.answer = "[none]"
                    placeholder_answers[node.id] = "[none]"
                    
                    node_answers[node.id] = "[none]"
                    return node_answers
                
                if "[answer_subquestion1]" in actual_question:
                    actual_question = actual_question.replace("[answer_subquestion1]", dependent_answer)
                elif f"[answer from {node.depends_on}]" in actual_question:
                    actual_question = actual_question.replace(f"[answer from {node.depends_on}]", dependent_answer)
                
                node.display_question = actual_question
            
            
            full_response = answer_question(
                question=actual_question,
                dataset=DATASET,
                method=METHOD,
                chunk_size=CHUNK_SIZE,
                min_sentence=MIN_SENTENCE,
                overlap=OVERLAP,
                topk1=TOPK1,
                topk2=TOPK2,
                max_iterations=MAX_ITERATIONS
            )
            answer = extract_answer(full_response)
            node.answer = answer
            placeholder_answers[node.id] = answer
            node_answers[node.id] = answer
            
            
            
            
            if (("[answer_subquestion1]" in node.question or "[answer from" in node.question) and 
                not updated_tree and node.answer and node.answer.lower() != "[none]"):
                
                
                if node.parent and not node.is_left_child:
                    print(f"node {node.id}: Contains the replacement tag and has obtained the answer. It needs to be rebuilt")
                    
            
            return node_answers
        
       
        left_answers = solve_node(node.left, updated_tree, current_depth + 1)
        node_answers.update(left_answers)
        
        
        needs_reconstruction = False
        if node.right and node.type == "Sequential":
            
            if ("[answer_subquestion1]" in node.right.question or 
                (node.right.depends_on and f"[answer from {node.right.depends_on}]" in node.right.question)):
                
               
                if node.left and node.left.id in placeholder_answers:
                    left_answer = placeholder_answers[node.left.id]
                   
                    if left_answer.lower() != "[none]":
                        needs_reconstruction = True
        
      
        if needs_reconstruction and not updated_tree and enhanced_right_subtree:
           
            
           
            new_right_question = generate_right_question_with_llm(
                parent_question=node.question,
                left_question=node.left.question if node.left else "",
                left_answer=placeholder_answers[node.left.id],
                original_right_question=node.right.question,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            
           
            
            
            remaining_height = max(max_height - current_depth - 1, 1) 
            
           
            new_right_node = build_enhanced_right_subtree(
                original_question=new_right_question,
                left_answer=placeholder_answers[node.left.id],
                api_url=api_url,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                examples_db=examples_db,
                num_examples=num_examples,
                max_height=remaining_height,  
                num_variants=right_subtree_variants,
                trees_per_variant=right_subtree_trees_per_variant
            )
            
            
            node.right = new_right_node
            
            
            
           
            right_answers = solve_node(node.right, True, current_depth + 1)
            node_answers.update(right_answers)
        elif needs_reconstruction and not updated_tree:
          
            
          
            new_right_question = generate_right_question_with_llm(
                parent_question=node.question,
                left_question=node.left.question if node.left else "",
                left_answer=placeholder_answers[node.left.id],
                original_right_question=node.right.question,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            
    
            
            
            remaining_height = max(max_height - current_depth - 1, 1)  
            
            
            new_right_node = build_question_tree(
                new_right_question, api_url, max_tokens, temperature, top_p,
                frequency_penalty, presence_penalty, examples_db, num_examples,
                depth=current_depth + 1, parent=node, is_left_child=False, 
                max_height=remaining_height, placeholder_answers=placeholder_answers
            )
            
          
            node.right = new_right_node
            
          

            right_answers = solve_node(node.right, True, current_depth + 1)
            node_answers.update(right_answers)
        else:
            
            if (node.right and ("[answer_subquestion1]" in node.right.question or 
                (node.right.depends_on and f"[answer from {node.right.depends_on}]" in node.right.question))):
                
               
                if node.left and node.left.id in placeholder_answers:
                    new_question = generate_right_question_with_llm(
                        parent_question=node.question,
                        left_question=node.left.question,
                        left_answer=placeholder_answers[node.left.id],
                        original_right_question=node.right.question
                    )
                    node.right.question = new_question
                    node.right.display_question = new_question
            
           
            right_answers = solve_node(node.right, updated_tree, current_depth + 1)
            node_answers.update(right_answers)
        
       
        child_questions = []
        valid_child_answers = False
        
       
        if node.left and node.left.id in node_answers:
            left_answer = node_answers[node.left.id]
            left_question = node.left.display_question
            child_questions.append((left_question, left_answer))
            
            if left_answer.lower() != "[none]":
                valid_child_answers = True
        
      
        if node.right and node.right.id in node_answers:
            right_answer = node_answers[node.right.id]
            right_question = node.right.display_question
            child_questions.append((right_question, right_answer))
           
            if right_answer.lower() != "[none]":
                valid_child_answers = True
        
       
        if node.type == "Sequential" and node.left and node.left.id in node_answers:
            if node_answers[node.left.id].lower() == "[none]":
                valid_child_answers = False
                
        
       
        if valid_child_answers and child_questions:
        
            final_answer = get_final_answer(node.display_question, child_questions, api_url)
            extracted_answer = re.search(r'final answer is:\s*(.*)', final_answer, re.DOTALL)
            if not extracted_answer:
                extracted_answer = re.search(r'answer is:\s*(.*)', final_answer, re.DOTALL)
            node.answer = extracted_answer.group(1).strip() if extracted_answer else final_answer
            placeholder_answers[node.id] = node.answer
            node_answers[node.id] = node.answer
            
            child_answers_str = ", ".join([f"「{q}」: 「{a}」" for q, a in child_questions])
           
            
            if node.answer.lower() == "[none]":
               
                full_response = answer_question(
                    node.display_question,
                    dataset=DATASET,
                    method=METHOD,
                    chunk_size=CHUNK_SIZE,
                    min_sentence=MIN_SENTENCE,
                    overlap=OVERLAP,
                    topk1=TOPK1,
                    topk2=TOPK2,
                    max_iterations=MAX_ITERATIONS
                )
                answer = extract_answer(full_response)
                node.answer = answer
                placeholder_answers[node.id] = answer
                node_answers[node.id] = answer
               
        else:

           
            full_response = answer_question(
                node.display_question,
                dataset=DATASET,
                method=METHOD,
                chunk_size=CHUNK_SIZE,
                min_sentence=MIN_SENTENCE,
                overlap=OVERLAP,
                topk1=TOPK1,
                topk2=TOPK2,
                max_iterations=MAX_ITERATIONS
            )
            answer = extract_answer(full_response)
            node.answer = answer
            placeholder_answers[node.id] = answer
            node_answers[node.id] = answer
           
        
        return node_answers
    
 
    all_answers = solve_node(root, False, 0)
    

    final_result = root.answer if root.answer else "[none]"
    
    return final_result


#---------------------------------------------------- Tree Statistics Collection -----------------------------------------------
def get_tree_statistics(root):
    """
    Calculate the statistics of a tree: height and node count
    Returns a tuple (height, node_count)
    """
    if root is None:
        return (0, 0)
    
    # Get all nodes
    all_nodes = get_all_nodes_postorder(root)
    node_count = len(all_nodes)
    
    # Calculate tree height
    def tree_height(node):
        if node is None:
            return 0
        return max(tree_height(node.left), tree_height(node.right)) + 1
    
    height = tree_height(root)
    
    return (height, node_count)

#---------------------------------------------------- Multi-Tree Multi-Variant Solution -----------------------------------------------
def decompose_and_answer_with_variants(question, trees_per_question=TREES_PER_QUESTION, api_url=None, max_tokens=MAX_TOKENS, 
                                     temperature=DECOMPOSE_TEMPERATURE, top_p=TOP_P, frequency_penalty=FREQUENCY_PENALTY, presence_penalty=PRESENCE_PENALTY, 
                                     num_examples=NUM_EXAMPLES, max_height=MAX_HEIGHT, enhanced_right_subtree=ENHANCED_RIGHT_SUBTREE,
                                     right_subtree_variants=RIGHT_SUBTREE_VARIANTS, right_subtree_trees_per_variant=RIGHT_SUBTREE_TREES_PER_VARIANT,
                                     max_variants=MAX_VARIANTS, stats_file_path=STATS_FILE_PATH):
    """
    Generate 6 trees per question, categorize by shape, and solve only one tree from the most common type.
    If unsuccessful, generate new question variants and try again.
    """
    global global_node_counter
    
    # Reset global counter to ensure each call starts from 0
    global_node_counter = 0
    
    examples_db = get_examples_database()
    
    # Set trees_per_question to 6 as requested
    trees_per_question = trees_per_question
    
    # Track attempt with original question and variants
    attempt_count = 0
    current_question = question
    attempted_questions = [question]  # Keep track of questions we've tried
    
    # Variables to track tree heights
    initial_height = 0
    final_height = 0
    success = False
    
    while attempt_count <= MAX_VARIANTS:
        
        print(f"\n{'='*80}")
        print(f"ATTEMPT {attempt_count+1} with question: '{current_question}'")
        print(f"{'='*80}")
        
        all_trees = []
        
        # Generate trees for the current question
        print(f"\nGenerating {trees_per_question} trees for current question (max height: {max_height})")
        
        for j in range(trees_per_question):
            # Use different temperature for tree diversity
            tree_temp = 0
            
            print(f"\nBuilding tree {j+1} for current question (temperature={tree_temp}, max_height={max_height}):")
            
            # Build the tree with height limitation
            root = build_question_tree(
                current_question, api_url, max_tokens, tree_temp, top_p, frequency_penalty,
                presence_penalty, examples_db, num_examples, depth=0,
                placeholder_answers={}, max_height=max_height
            )
            
            # Get tree statistics
            height, node_count = get_tree_statistics(root)
            
            # Save tree information
            all_trees.append({
                'tree': root,
                'tree_num': j+1,
                'height': height,
                'node_count': node_count,
                'question_text': current_question
            })
            
            print(f"Tree {j+1} - Height: {height}, Node count: {node_count}")
        
        # Calculate tree shape frequencies
        tree_shape_counter = Counter()
        for tree_info in all_trees:
            shape = (tree_info['height'], tree_info['node_count'])
            tree_shape_counter[shape] += 1
        
        # Print shape frequencies
        print("\nTree shape frequencies (height, node count):")
        for shape, count in tree_shape_counter.most_common():
            print(f"Height: {shape[0]}, Node count: {shape[1]} - Frequency: {count}")
        
        # Get the most common shape
        if tree_shape_counter:
            most_common_shape, most_common_count = tree_shape_counter.most_common(1)[0]
            print(f"\nMost common shape: Height {most_common_shape[0]}, Node count {most_common_shape[1]} (Count: {most_common_count})")
            
            # Filter trees to only include the most common shape
            most_common_trees = [tree_info for tree_info in all_trees 
                               if (tree_info['height'], tree_info['node_count']) == most_common_shape]
            
            print(f"Found {len(most_common_trees)} trees with the most common shape")
            
            # Only solve the first tree from the most common shape
            if most_common_trees:
                tree_info = most_common_trees[0]
                tree_root = tree_info['tree']
                question_text = tree_info['question_text']
                
                # Save initial height
                initial_height = tree_info['height']
                
                print(f"\n{'-'*80}")
                print(f"Attempting to solve one tree from most common shape: Tree {tree_info['tree_num']}")
                print(f"Question: '{question_text}'")
                print(f"Tree height: {tree_info['height']}, Node count: {tree_info['node_count']}")
                print(f"{'-'*80}")
                
                print("\nTree structure:")
                print_all_nodes(tree_root)
                
                # Create a new placeholder_answers dictionary for each tree to avoid confusion
                placeholder_answers = {}
                
                answer = solve_tree(
                    tree_root, question_text, api_url, max_tokens, temperature, top_p,
                    frequency_penalty, presence_penalty, examples_db, num_examples,
                    enhanced_right_subtree=enhanced_right_subtree,
                    right_subtree_variants=right_subtree_variants,
                    right_subtree_trees_per_variant=right_subtree_trees_per_variant,
                    max_height=max_height,
                    placeholder_answers=placeholder_answers
                )
                
                # Calculate the final height after solving (which might have expanded the tree)
                final_height, _ = get_tree_statistics(tree_root)
                
                print(f"\nSelected tree returned answer: '{answer}'")
                print(f"Initial tree height: {initial_height}, Final tree height after solving: {final_height}")
                
                # If we found a valid (non-[none]) answer, use it and stop
                if answer.lower() != "[none]":
                    print(f"Found valid answer, stopping tree traversal")
                    print("\n" + "="*80)
                    print(f"Final answer for question: '{question}'")
                    print(f"Answer: '{answer}'")
                    print("="*80)
                    
                    # Save tree statistics
                    success = True
                    save_tree_stats(question, answer, initial_height, final_height, stats_file_path, success)
                    
                    return answer
                else:
                    print(f"Tree returned [none], will try with a new question variant")
        
        # Generate question variants
        attempt_count += 1
        if attempt_count <= 2:
            print(f"\n{'-'*80}")
            print(f"No valid answer found from selected tree. Generating new question variant {attempt_count}")
            print(f"{'-'*80}")
            
            # Use the existing generate_question_variants function
            new_variants = generate_question_variants(question, num_variants=1)
            
            # Get the new variant (skipping the first one which is the original question)
            if len(new_variants) > 1:
                new_question = new_variants[1]
            else:
                print(f"Warning: generate_question_variants failed to produce a variant, falling back to original question")
                new_question = question
            
            print(f"New question variant {attempt_count}: '{new_question}'")
            current_question = new_question
            attempted_questions.append(current_question)
        else:
            # If we've exhausted all variants, use direct lookup on the original question
            print(f"\n{'-'*80}")
            print("Exhausted all variants. Using direct lookup on original question.")
            print(f"{'-'*80}")
            
            # Directly call answer_question on the original question
            final_answer = direct_answer(question, dataset=DATASET,
                method=METHOD,
                chunk_size=CHUNK_SIZE,
                min_sentence=MIN_SENTENCE,
                overlap=OVERLAP,
                topk1=TOPK1,
                topk2=TOPK2,
            )
            
            # Save tree statistics with direct lookup
            success = False
            save_tree_stats(question, final_answer, initial_height, final_height, stats_file_path, success)
            
            return final_answer
    
    # If we somehow get here (should not happen with the logic above)
    # Save tree statistics as a failure
    success = False
    save_tree_stats(question, "Could not determine an answer", initial_height, final_height, stats_file_path, success)
    
    return "Could not determine an answer after trying original question and variants."



def main():
    """
    Main function that directly calls the question decomposition and answering functionality
    """
    # Set your question here or get user input at runtime

    # Who is the son of the Italian navigator who explored the eastern coast of the continent Ulises Solís' birthplace is located in for England?
    question = """
    为英国探索了尤利西斯·索利斯出生地所在大陆东海岸的那位意大利航海家，他的儿子是谁？
    """
    
    # Set maximum tree height
    max_height = 1
    
    # Set enhanced right subtree parameters
    enhanced_right_subtree = True
    right_subtree_variants = 2
    right_subtree_trees_per_variant = 3
    
    # Set file path for statistics
    stats_file_path = ""
    
    answer = decompose_and_answer_with_variants(
        question=question,
        trees_per_question=1,
        api_url=None,
        max_tokens=9000,
        temperature=0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        num_examples=25,
        max_height=max_height,
        enhanced_right_subtree=enhanced_right_subtree,
        right_subtree_variants=right_subtree_variants, 
        right_subtree_trees_per_variant=right_subtree_trees_per_variant,  
        max_variants=0,
        stats_file_path=stats_file_path
    ) 
     
    # Print final result       
    print(f"\nFinal answer: {answer}")        
 
# Run the main function directly 
if __name__ == "__main__":        
    main()   