import numpy as np
import pandas as pd
import pickle
import sys
import os
import time
import argparse
from tqdm.auto import tqdm
from typing import List, Dict, Any, Tuple
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
from routellm import RouteLLMRouter
load_dotenv()

DIFFICULTY_DB_PATH = "faiss_difficulty_db"
RESPONSE_DBS_DIR = "faiss_response_dbs"

# LLM_LIST = [
#     "WizardLM/WizardLM-13B-V1.2",
#     "claude-instant-v1",
#     "claude-v1",
#     "claude-v2",
#     "gpt-3.5-turbo-1106",
#     "gpt-4-1106-preview",
#     "meta/code-llama-instruct-34b-chat",
#     "meta/llama-2-70b-chat",
#     "mistralai/mistral-7b-chat",
#     "mistralai/mixtral-8x7b-chat",
#     "zero-one-ai/Yi-34B-Chat",
# ]
LLM_LIST = {
    'WizardLM/WizardLM-13B-V1.2': 1.03, 
    'claude-instant-v1': 1.23, 
    'claude-v1': 3.59, 
    'claude-v2': 3.93, 
    'gpt-3.5-turbo-1106': 1.24, 
    'gpt-4-1106-preview': 5.0, 
    'meta/code-llama-instruct-34b-chat': 1.16, 
    'meta/llama-2-70b-chat': 1.19, 
    'mistralai/mistral-7b-chat': 1.0, 
    'mistralai/mixtral-8x7b-chat': 1.11, 
    'zero-one-ai/Yi-34B-Chat': 1.17
}

config_list = [
    {
        "name": "qwen2.5",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "api_key": "NULL",
        "base_url": "http://0.0.0.0:8000/v1",
    },
    {
        "name": "gpt-4o-mini",
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY1"),
    },
    {
        "name": "gemma3",
        "model": "google/gemma-3-4b-it",
        "api_key": os.getenv("GEMINI_API_KEY10"),
        # "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "base_url": "http://0.0.0.0:8000/v1"
    },
    {
        "name": "qwen3",
        "model": "Qwen/Qwen3-4B",
        "api_key": "NULL",
        "base_url": "http://0.0.0.0:8000/v1",
    },
    {
        "name": "gemini-2.5-flash-lite",
        "model": "gemini-2.5-flash-lite-preview-06-17",
        "api_key": os.getenv('GEMINI_API_KEY'),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    {
        "name": "o3",
        "model": "o3",
        "api_key": os.getenv('OPENAI_API_KEY'),
    }
]

llm_config = {
    "config_list": config_list,
    "cache_seed": 42,
    "temperature": 0.5,
}

# This will be initialized in main after parsing args
client = None
agent_model_name = None
temperature = None

try:
    with open('llm_analyses_results.pkl', 'rb') as f:
        response_data = pickle.load(f)
except (FileNotFoundError, EOFError):
    print("Warning: 'llm_analyses_results.pkl' not found. Response retrieval will be disabled.")
    response_data = {}

try:
    difficulty_df = pd.read_csv('sampled_router_bench_with_difficulty_analysis.csv')
    difficulty_data = difficulty_df.set_index('sample_id')['difficulty_analysis_summary'].to_dict()
except (FileNotFoundError, KeyError):
    print("Warning: 'sampled_router_bench_with_difficulty_analysis.csv' not found. Difficulty retrieval will be disabled.")
    difficulty_data = {}

class DifficultyAnalystAgent:
    def __init__(self, client: OpenAI, model_name: str, temperature: float):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.system_message = """Your role as an assistant is to analyze the difficulty of a given query for a large language model through a systematic long thinking process analysis. You will be provided with the user query and some context from past similar analyses. You need to evaluate the incoming query on several key dimensions: reasoning, comprehension, instruction following, agentic, knowledge retrieval, coding, multilingual. For each dimension, elaborate on the specific challenges and required capabilities. Now, try to analyze the following query through the above guidelines:\n"""
        
    def analyze(self, query: str, relevant_analyses: List[Any]) -> Dict[str, Any]:
        formatted_analyses = "\n".join([f"- {doc.page_content}" for doc in relevant_analyses])

        prompt = f"""
        **Context from similar past analyses:**
        {formatted_analyses if formatted_analyses else "No relevant past analyses found."}

        ---

        **Query to Analyze:**
        "{query}"
        """
        
        while True:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    # {"role": "system", "content": self.system_message},
                    {"role": "user", "content": self.system_message + prompt},
                ],
                temperature=self.temperature,
                max_tokens=2000,
                # frequency_penalty=1.0,
            )
            if response.choices[0].message.content is not None:
                difficulty_assessment = response.choices[0].message.content.strip().lower()
                return {
                    "query": query,
                    "difficulty": difficulty_assessment,
                    "analysis": f"The query is classified as '{difficulty_assessment}' based on LLM analysis."
                }

class Retriever:
    def __init__(self, difficulty_db: FAISS, response_dbs: Dict[str, FAISS]):
        self.difficulty_db = difficulty_db
        self.response_dbs = response_dbs

    def retrieve_difficulty_analyses(self, query: str, k: int = 3) -> List[Any]:
        if self.difficulty_db:
            return self.difficulty_db.similarity_search(query, k=k)
        return []

    def retrieve_model_responses(self, query: str, k: int = 3) -> Dict[str, List[Any]]:
        all_responses = {}
        if self.response_dbs:
            for model_name, db in self.response_dbs.items():
                all_responses[model_name] = db.similarity_search(query, k=k)
        return all_responses

class RoutingDecisionMakerAgent:
    def __init__(self, client: OpenAI, model_name: str, temperature: float):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.system_message = """You are an intelligent routing decision maker for a multi-agent system.
Your task is to identify all AI models that can correctly answer the given query.
You will be provided with the user's query, a difficulty analysis of that query, and several retrieved examples of past model responses.
Based on the provided information, especially the past responses, identify all models that you believe can successfully answer the query.
Your final answer must be a comma-separated list of the names of the chosen models (e.g., 'Model-A, Model-C').\n"""

    def decide(self, query: str, difficulty_analysis: Dict[str, Any], relevant_responses: Dict[str, List[Any]], llm_list: Dict[str, float]) -> List[str]:
        # Create a mapping to anonymize model names to mitigate bias
        llm_names = list(llm_list.keys())
        anonymized_names = [f"Model-{chr(65+i)}" for i in range(len(llm_names))]
        anonymized_map = dict(zip(llm_names, anonymized_names))
        reverse_anonymized_map = dict(zip(anonymized_names, llm_names))

        # Anonymized model list for the prompt
        available_models_str = ', '.join(anonymized_names)

        formatted_responses = ""
        for model_name, docs in relevant_responses.items():
            anonymized_name = anonymized_map.get(model_name, model_name)
            formatted_responses += f"\n\n**Retrieved Responses for {anonymized_name}:**\n"
            if docs:
                formatted_responses += "\n".join([f"  - {doc.page_content}" for doc in docs])
            else:
                formatted_responses += "  No relevant responses found for this model."

        prompt = f"""
        **Query to Route:**
        "{query}"

        **Difficulty Analysis of the Query:**
        {difficulty_analysis['difficulty']}

        ---

        **Retrieved Similar Examples for Context:**
        
        **Relevant Past Model Response Analyses (per model):**
        {formatted_responses if formatted_responses else "No relevant model responses found."}

        ---
        
        **Decision Task:**
        Based on the difficulty analyses and the models' demonstrated capabilities from past responses, which models can correctly answer the given query?
        
        The available models are: {available_models_str}
        
        Your final answer must be a comma-separated list of capable models from the list above (e.g., Model-A, Model-C).
        """
        
        loop_cnt = 0
        while True:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    # {"role": "system", "content": self.system_message},
                    {"role": "user", "content": self.system_message + prompt},
                ],
                temperature=self.temperature,
                max_tokens=2000,
                # frequency_penalty=1.0,
            )
            if response.choices[0].message.content is not None:
                decision_str = response.choices[0].message.content.replace('TERMINATE', '').strip()
                potential_candidates = [name.strip() for name in decision_str.split(',')]
                
                candidate_llms = []
                for name in potential_candidates:
                    if name in reverse_anonymized_map:
                        candidate_llms.append(reverse_anonymized_map[name])
                
                if candidate_llms:
                    return candidate_llms
                else:
                    print(f"Attempt {loop_cnt+1} Invalid decision '{decision_str}'. No valid models found. Please choose from: {', '.join(anonymized_names)}")
                    loop_cnt += 1
                    if loop_cnt >= 10:
                        return [np.random.choice(llm_names)]  # Fallback to random choice
            # else:
            #     loop_cnt += 1
                

class AgenticRouter:
    def __init__(self, retriever: Retriever, llm_list: Dict[str, float], client: OpenAI, model_name: str, temperature: float):
        self.difficulty_analyst = DifficultyAnalystAgent(client, model_name, temperature)
        self.retriever = retriever
        self.decision_maker = RoutingDecisionMakerAgent(client, model_name, temperature)
        self.llm_list = llm_list

    def route(self, query: str) -> Tuple[str, Dict[str, Any]]:
        # Step 1: Retrieve relevant difficulty analyses to provide context to the analyst.
        relevant_analyses = self.retriever.retrieve_difficulty_analyses(query)

        # Step 2: Analyze query difficulty with the context of past analyses.
        difficulty_analysis = self.difficulty_analyst.analyze(query, relevant_analyses)
        
        # Step 3: Retrieve relevant model responses for the decision maker.
        relevant_responses = self.retriever.retrieve_model_responses(difficulty_analysis['difficulty'])
        
        # Step 4: Get candidate models from decision maker.
        candidate_models = self.decision_maker.decide(
            query,
            difficulty_analysis,
            relevant_responses,
            self.llm_list
        )

        # Step 5: Select the lowest-cost model from the candidates.
        if not candidate_models:
            # Fallback if decision_maker returns an empty list (should be handled by its retry logic, but as a safeguard)
            decision = np.random.choice(list(self.llm_list.keys()))
        else:
            # Sort candidates by cost and pick the first one
            candidate_models.sort(key=lambda model_name: self.llm_list.get(model_name, float('inf')))
            decision = candidate_models[0]

        return decision, difficulty_analysis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Agentic Router Benchmark")
    parser.add_argument(
        "--model-config",
        type=str,
        default="qwen2.5",
        choices=[c["name"] for c in config_list],
        help="The name of the model configuration to use for the agent."
    )
    parser.add_argument(
        "--routing-method",
        type=str,
        default="agentic",
        choices=["agentic", "routellm"],
        help="The routing method to use: 'agentic' or 'routellm'"
    )
    args = parser.parse_args()

    # Initialize OpenAI from selected config
    selected_config = next((c for c in config_list if c["name"] == args.model_config), None)
    if not selected_config:
        print(f"Error: Model configuration '{args.model_config}' not found.")
        sys.exit(1)
    
    print(f"Using model configuration: '{selected_config['name']}'")
    client = OpenAI(
        api_key=selected_config.get("api_key"),
        base_url=selected_config.get("base_url")
    )
    agent_model_name = selected_config["model"]
    temperature = llm_config["temperature"]

    print("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'cuda:1'}
    )
    print("Embedding model initialized on cuda:1.")

    # --- Load data and split into train/test for benchmark ---
    print("Loading and splitting data for benchmark...")
    try:
        full_df = pd.read_csv('router_bench_with_keywords.csv')
        train_df = pd.read_csv('sampled_router_bench_with_difficulty_analysis.csv')
        
        train_ids = set(train_df['sample_id'])
        test_df = full_df[~full_df['sample_id'].isin(train_ids)].copy()
        
        # The training data for difficulty analysis is from train_df
        difficulty_data = train_df.set_index('sample_id')['difficulty_analysis_summary'].to_dict()
        print(f"Loaded {len(train_df)} samples for training and {len(test_df)} for testing.")
        
        # --- Stratified sampling of test_df ---
        total_test_samples = 2000
        n_per_category = (test_df['eval_name'].value_counts(normalize=True) * total_test_samples).round().astype(int)
        diff = total_test_samples - n_per_category.sum()
        if diff != 0:
            n_per_category[n_per_category.idxmax()] += diff
        test_df = test_df.groupby('eval_name', group_keys=False).apply(
            lambda x: x.sample(n=int(n_per_category[x.name]), random_state=42)
        ).reset_index(drop=True)
        print(f"Final test set sampled to {len(test_df)} samples based on eval_name proportion.")
    except FileNotFoundError as e:
        print(f"Error: Required data file not found: {e}. Aborting benchmark.")
        sys.exit(1)

    difficulty_db = None
    if os.path.exists(DIFFICULTY_DB_PATH):
        print(f"Loading difficulty DB from {DIFFICULTY_DB_PATH}...")
        difficulty_db = FAISS.load_local(
            DIFFICULTY_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print("Difficulty DB loaded.")
    elif difficulty_data:
        print("Creating difficulty analysis vector database from training data...")
        difficulty_documents = [
            Document(page_content=summary, metadata={"sample_id": sample_id})
            for sample_id, summary in difficulty_data.items()
        ]
        difficulty_db = FAISS.from_documents(difficulty_documents, embedding_model)
        difficulty_db.save_local(DIFFICULTY_DB_PATH)
        print(f"Difficulty DB saved to {DIFFICULTY_DB_PATH}.")

    # --- Load or Create Response DBs ---
    response_dbs = {}
    if os.path.exists(RESPONSE_DBS_DIR):
        print(f"Loading response DBs from {RESPONSE_DBS_DIR}...")
        for model_name in LLM_LIST:
            model_db_path = os.path.join(RESPONSE_DBS_DIR, model_name)
            if os.path.exists(model_db_path):
                print(f"  - Loading DB for {model_name}...")
                response_dbs[model_name] = FAISS.load_local(
                    model_db_path, 
                    embedding_model, 
                    allow_dangerous_deserialization=True
                )
        print("Response DBs loaded.")
    elif response_data:
        print("Creating and saving separate vector stores for each model's responses...")
        model_documents = {}
        for model_name, responses in response_data.items():
            if isinstance(responses, list):
                model_documents[model_name] = [
                    Document(page_content=str(response), metadata={"model": model_name})
                    for response in responses
                ]
        os.makedirs(RESPONSE_DBS_DIR, exist_ok=True)
        for model_name, docs in model_documents.items():
            if docs:
                print(f"  - Creating and saving vector store for {model_name}...")
                model_db_path = os.path.join(RESPONSE_DBS_DIR, model_name)
                db = FAISS.from_documents(docs, embedding_model)
                db.save_local(model_db_path)
                response_dbs[model_name] = db
                print(f"  - Vector store for {model_name} saved to {model_db_path}.")

    if not difficulty_db and not response_dbs:
        print("\nError: No data available to create vector databases. Please check your data files.")
        sys.exit(1)

    print("\nInitializing Router...")
    if args.routing_method == "agentic":
        retriever = Retriever(difficulty_db, response_dbs)
        router = AgenticRouter(retriever, LLM_LIST, client, agent_model_name, temperature)
        print("Agentic Router initialized successfully.")
    elif args.routing_method == "routellm":
        router = RouteLLMRouter()
        try:
            router.load_model()
            print("RouteLLM Router initialized successfully.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please train the RouteLLM model first using: python routellm.py --mode train")
            sys.exit(1)

    print("\n" + "="*50 + "\n")
    print("Starting benchmark on the test set...")

    total_correct = 0
    total_cost = 0.0
    num_test_samples = len(test_df)
    difficulty_analyses_results = []
    routing_decisions_results = []

    if num_test_samples > 0:
        for index, row in tqdm(test_df.iterrows(), total=num_test_samples, desc="Benchmarking"):
            query = row['prompt']

            if args.routing_method == "agentic":
                decision, difficulty_analysis = router.route(query)
                difficulty_analyses_results.append(difficulty_analysis['difficulty'])
            elif args.routing_method == "routellm":
                decision = router.route(query)
                difficulty_analyses_results.append("N/A")  # RouteLLM doesn't do difficulty analysis
                
            routing_decisions_results.append(decision)

            # Check correctness
            if decision in test_df.columns:
                correctness = row[decision]
                total_correct += correctness

            # Calculate cost
            cost_column = f"{decision}|total_cost"
            if cost_column in test_df.columns:
                cost = row[cost_column]
                total_cost += cost

        test_df['agent_difficulty_analysis'] = difficulty_analyses_results
        test_df['agent_routing_decision'] = routing_decisions_results
        
        results_filename = f"{args.routing_method}_router_routerbenchsample{total_test_samples}_results_{args.model_config}.csv"
        test_df.to_csv(results_filename, index=False)
        print(f"\nBenchmark results with analyses and decisions saved to {results_filename}")

        print("\n" + "="*50)
        print("Benchmark Finished!")
        print(f"Total test samples: {num_test_samples}")
        print(f"Total correct decisions: {total_correct}")
        print(f"Total cumulative cost (Router): {total_cost}")

        # Calculate and print cost if all queries were routed to gpt-4-1106-preview
        gpt4_cost_column = "gpt-4-1106-preview|total_cost"
        total_gpt4_cost = 0.0
        if gpt4_cost_column in test_df.columns:
            total_gpt4_cost = test_df[gpt4_cost_column].sum()
            print(f"Total cumulative cost (All to gpt-4-1106-preview): {total_gpt4_cost}")
        else:
            print(f"Warning: Cost column '{gpt4_cost_column}' not found in test data. Cannot calculate baseline cost.")

        # Calculate and print accuracy if all queries were routed to gpt-4-1106-preview
        gpt4_accuracy_column = "gpt-4-1106-preview"
        total_gpt4_correct = 0
        if gpt4_accuracy_column in test_df.columns:
            total_gpt4_correct = test_df[gpt4_accuracy_column].sum()
            print(f"Total correct decisions (All to gpt-4-1106-preview): {total_gpt4_correct}")
        else:
            print(f"Warning: Accuracy column '{gpt4_accuracy_column}' not found in test data. Cannot calculate baseline accuracy.")

        # Calculate and print for random routing baseline
        np.random.seed(42) # for reproducibility
        random_total_correct = 0.0
        random_total_cost = 0.0
        llm_names = list(LLM_LIST.keys())
        for _, row in test_df.iterrows():
            random_decision = np.random.choice(llm_names)
            if random_decision in test_df.columns:
                random_total_correct += row[random_decision]
            
            random_cost_column = f"{random_decision}|total_cost"
            if random_cost_column in test_df.columns:
                random_total_cost += row[random_cost_column]
        
        print(f"Total correct decisions (Random): {random_total_correct}")
        print(f"Total cumulative cost (Random): {random_total_cost}")


        print("="*50 + "\n")
    else:
        print("No test samples found to benchmark.")
