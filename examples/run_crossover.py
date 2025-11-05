import os

import ioh

from iohblade.experiment import Experiment
from iohblade.llm import Gemini_LLM, Ollama_LLM, OpenAI_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA, RandomSearch
from iohblade.problems import BBOB_SBOX

if __name__ == "__main__":  # prevents weird restarting behaviour
    experiment_name = "BBOB"
    
    api_key_gemini = os.getenv("GEMINI_API_KEY")

    llm = Gemini_LLM(
        api_key_gemini, "gemini-2.0-flash"
    )
    budget = 50  # short budgets

    mutation_prompts = [
        "Refine the selected algorithm to improve it."
    ]

    for llm in [llm]:  # , llm2, llm3, llm4, llm5, llm2
        RS_method = RandomSearch(llm, budget=budget)

        LLaMEA_method1 = LLaMEA(
            llm,
            budget=budget,
            name=f"LLaMEA-1",
            mutation_prompts=mutation_prompts,
            n_parents=4,
            n_offspring=12,
            elitism=False,
            mutation_rate=0.1
        )
        LLaMEA_method2 = LLaMEA(
            llm,
            budget=budget,
            name=f"LLaMEA-2",
            mutation_prompts=mutation_prompts,
            n_parents=4,
            n_offspring=12,
            elitism=False,
            mutation_rate=0.3
        )
        LLaMEA_method3 = LLaMEA(
            llm,
            budget=budget,
            name=f"LLaMEA-3",
            mutation_prompts=mutation_prompts,
            n_parents=4,
            n_offspring=12,
            elitism=False,
            mutation_rate=1
        )

        methods = [
            LLaMEA_method1,
            LLaMEA_method2,
            LLaMEA_method3,
            RS_method
        ]

        # List containing function IDs we consider
        fids = [1, 3, 6, 8, 10, 13, 15, 17, 21, 23]

        training_instances = [(f, i) for f in fids for i in range(1, 6)]
        test_instances = [(f, i) for f in fids for i in range(5, 16)]

        log_dir = f"results/{experiment_name}"
        os.makedirs(log_dir, exist_ok=True)
        logger = ExperimentLogger(log_dir)

        problems = []
        problems.append(
            BBOB_SBOX(
                training_instances=training_instances,
                test_instances=test_instances,
                dims=[5],
                budget_factor=2000,
                eval_timeout=600,
                name=f"BBOB",
                problem_type=ioh.ProblemClass.BBOB,
                full_ioh_log=True,
                ioh_dir=f"{logger.dirname}/ioh",
            )
        )

        experiment = Experiment(
            methods=methods,
            problems=problems,
            runs=3,
            show_stdout=False,
            exp_logger=logger,
            budget=budget,
        )  # normal run
        experiment()  # run the experiment
