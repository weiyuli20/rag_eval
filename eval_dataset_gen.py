from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import LLM
from dotenv import load_dotenv
import os
import random
from tqdm import tqdm
from typing import List
from langchain_core.documents.base import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import json
# from prompt import question_relevance_critique_prompt, question_standalone_critique_prompt, question_groundedness_critique_prompt, QA_generation_prompt
from prompt_zh import question_relevance_critique_prompt, question_standalone_critique_prompt, question_groundedness_critique_prompt, QA_generation_prompt

load_dotenv()  


llm_client = ChatOpenAI(
    base_url="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen2.5-14B-Instruct",
    api_key=os.environ.get("api_key"),
    temperature=0.2
)


critic_llm_client = ChatOpenAI(
    base_url="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen2.5-32B-Instruct",
    api_key=os.environ.get("api_key"),
    temperature=0.2
)



def call_llm(llm_client: LLM, prompt: str):
    response = llm_client.invoke(prompt)
    return response.content


# res = call_llm(llm_client, "This is a test context")
# print(res)



def generate_qa_dataset(llm_client,docs_processed:List[LangchainDocument]):

    N_GENERATIONS = 3  # We intentionally generate only 10 QA couples here for cost and time considerations

    print(f"Generating {N_GENERATIONS} QA couples...")

    outputs = []
    for sampled_context in tqdm(random.sample(docs_processed, N_GENERATIONS)):
        # Generate QA couple
        output_QA_couple = call_llm(llm_client, QA_generation_prompt.format(context=sampled_context.page_content))
        try:
            question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
            answer = output_QA_couple.split("Answer: ")[-1]
            assert len(answer) < 500, "Answer is too long"
            outputs.append(
                {
                    "context": sampled_context.page_content,
                    "question": question,
                    "answer": answer,
                    "source_doc": sampled_context.metadata["source"],
                }
            )
        except:
            continue
    
    return outputs


# 设置问题评估裁判


def critic_qa(llm_client, prompt, qa_list):
    print("Generating critique for each QA couple...")
    for output in tqdm(qa_list):
        evaluations = {
            "groundedness": call_llm(
                llm_client,
                question_groundedness_critique_prompt.format(context=output["context"], question=output["question"]),
            ),
            "relevance": call_llm(
                llm_client,
                question_relevance_critique_prompt.format(question=output["question"]),
            ),
            "standalone": call_llm(
                llm_client,
                question_standalone_critique_prompt.format(question=output["question"]),
            ),
        }
        try:
            for criterion, evaluation in evaluations.items():
                score, eval = (
                    int(evaluation.split("Total rating: ")[-1].strip()),
                    evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                )
                output.update(
                    {
                        f"{criterion}_score": score,
                        f"{criterion}_eval": eval,
                    }
                )
        except Exception as e:
            continue

    return qa_list



def filter_qa(generated_questions):
    print("Filtering QA couples...")
    generated_questions = pd.DataFrame(generated_questions)
    generated_questions = generated_questions.loc[
    (generated_questions["groundedness_score"] >= 1)
    & (generated_questions["relevance_score"] >= 1)
    & (generated_questions["standalone_score"] >= 1)
]
    
    return generated_questions



def load_docs(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ 文档文件夹不存在：{data_dir}")
    
    md_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith("_notable.md"):
                md_files.append(os.path.join(root, f))
    
    if not md_files:
        raise ValueError(f"❌ 无 md 文件：{data_dir}")
    
    # 读取文档内容
    raw_docs = []
    for path in tqdm(md_files, desc="📄 加载文档"):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        raw_docs.append(LangchainDocument(
            page_content=content,
            metadata={"source": os.path.basename(path).replace("_notable","")}  # 记录文件名作为来源
        ))

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            separators=["#","![]","\n\n", "\n",".", " ", ""],
        )
    doc_processed = []
    for doc in raw_docs:
        doc_processed += text_splitter.split_documents([doc])

    print(f"✅ 共处理 {len(doc_processed)} 个文档块")
    return doc_processed

if __name__ =='__main__':
    docs_processed=load_docs("../app/data/after_process")
    # qa_pairs=generate_qa_dataset(llm_client,docs_processed)
    qa_pairs=generate_qa_dataset(llm_client,docs_processed)
    with open("raw_qa_pairs","w",encoding="utf-8") as f:
        json_str = json.dumps(qa_pairs, ensure_ascii=False, indent=2)
        f.write(json_str)
    qa_list = critic_qa(critic_llm_client,question_groundedness_critique_prompt,qa_pairs)
    eval_dataset = filter_qa(qa_list)
    with open("final_eval_datasets","w",encoding="utf-8") as f:
        eval_dataset.to_json(f,orient="records",force_ascii=False,indent=2)
