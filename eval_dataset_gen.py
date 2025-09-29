from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import LLM
from dotenv import load_dotenv
import os
import random
from tqdm import tqdm
from typing import List
from langchain_core.documents.base import Document as LangchainDocument


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


QA_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""



def generate_qa_dataset(llm_client,docs_processed:List[LangchainDocument]):

    N_GENERATIONS = 10  # We intentionally generate only 10 QA couples here for cost and time considerations

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

question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

question_standalone_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independent from the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """



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
    generated_questions = generated_questions.loc[
    (generated_questions["groundedness_score"] >= 4)
    & (generated_questions["relevance_score"] >= 4)
    & (generated_questions["standalone_score"] >= 4)
]
    
    with open("final_eval_datasets","w",encoding="utf-8") as f:
        f.write(generated_questions)


if __name__ =='__main__':
    docs_processed=[]
    qa_pairs=generate_qa_dataset(llm_client,docs_processed)
    qa_list = critic_qa(critic_llm_client,question_groundedness_critique_prompt,qa_pairs)
    filter(qa_list)