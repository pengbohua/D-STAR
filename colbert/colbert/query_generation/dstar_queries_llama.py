"""This basic script demonstrates how to progressively write the generation output.

It assumes the generated language uses spaces to separate words.
"""

import os
import ctranslate2
import sentencepiece as spm
import json
import random
import argparse
import logging
import re


def white_space_fix(text):
    return ' '.join(text.split())


def filter_results(text):
    end_tokens = ['\?', '\n', 'Background', 'Mention']
    flag = False
    for eos in end_tokens:
        if eos in text:
            flag = True
    if not flag:
        return white_space_fix(text)
    pattern = '|'.join(map(re.escape, end_tokens))
    regex = re.compile(rf'.*?(?={pattern})', re.IGNORECASE | re.DOTALL)
    return white_space_fix(regex.search(text).group(0).strip())


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler to write the log to a file
log_file = 'logs/log.txt'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter for the log message
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger object
logger.addHandler(file_handler)
logger.info('Starting query generation...')



# Unbiased (not just supportive but also contradict) demonstrations provided by medical experts
DEMO_DICT = {
"Disease_disorder": ["Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question.",
       "Mention: chronic kidney disease\n Background: 40mg/day dosage of folic acid does not affect chronic kidney disease (CKD) progression\n Answer:Treatment with high doses of folic acid did not reduce the incidence of vascular disease in patients with advanced chronic kidney disease.",
        "Question: What is the impact of folic acid on the progression of chronic kidney disease?",
        "Mention: main hiv\n Background: Having a main partner worsens HIV outcomes.\n Answer: In an analysis stratified by previous antiretroviral therapy and clinical stage when starting HAART (US Centers for Disease Control), the adjusted hazard ratio for progression to AIDS was 0.79 for participants with a stable partnership compared with those without.",
        "Question: What kind of HIV outcomes will having a main partner lead to from some government statistics?",
                     ],
'Biological_structure': ["Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question.",
        "Mention: liver methadone\n Background: 32% of liver transplantation programs required patients to discontinue methadone treatment in 2001.\n Answer: Policies requiring discontinuation of methadone in 32% of all programs contradict the evidence base for efficacy of long-term replacement therapies.",
        "Question: What impact do policies which require discontinuation of methadone treatment have on the efficacy of liver transplant programs?",
        "Mention: breast cancer genetic factors\n Background: Breast cancer development is determined exclusively by genetic factors.\n Answer: Risks of breast cancer associated with low-penetrance susceptibility polymorphisms do not vary significantly with these ten established environmental risk factors.",
        "Question: Is there a correlation between breast cancer and genetic factors?",],

'Diagnostic_procedure': ["Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question.",
       "Mention: bias in the phage genome locations\n Background: A strong bias in the phage genome locations where the spacers were derived has been observed in many CRISPR subtypes that confer the immunity to phage\n Answer: We detect a strong and reproducible bias in the phage genome locations from which spacers derive",
        "Question: Which level of bias is observed in the phage genome locations?",

        "Mention: TDP-43 respiratory proteins neuronal loss\n Background: Blocking the interaction between TDP-43 and respiratory complex I proteins leads to increased neuronal loss.\n Answer: The suppression of TDP-43 mitochondrial localization abolishes mutant TDP-43-induced neuronal loss",
        "Question: What is the impact of blocking TDP-43 and respiratory proteins on neuronal loss?",
                         ],

'Detailed_description': ["Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question.",
        "Mention: chronic tension type headache\n Background: Amitriptyline is an effective treatment for chronic tension-type headaches.\n Answer: Tricyclic antidepressant medication produced larger reductions in headache activity, analgesic medication use, and headache-related disability than placebo",
        "Question: What is the efficacy of antidepressants for treating chronic tension-type headaches?",
        "Mention: infection rate transplantation of mesenchymal stem cells induction therapy\n Background: Autologous transplantation of mesenchymal stem cells causes a higher rate of opportunistic infections than induction therapy with anti-interleukin-2 receptor antibodies.\n Answer: the use of autologous MSCs compared with antibody induction therapy resulted in decreased risk of infection",
        "Question: Which will cause a higher infection rate, transplantation of mesenchymal stem cells or induction therapy?",
                         ],

'Medication':["Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question.",
        "Mention: antidepressants\n Background: Antidepressants reduce the severity of migraines\n Answer: Tricyclics significantly reduced the number of days with tension-type headache and number of headache attacks from migraine than placebo",
        "Question: How effective are antidepressants for treating migraines?",
        "Mention: Tirasemtiv muscle\n Background: Tirasemtiv has no effect on fast-twitch muscle.\n Answer: We developed a small-molecule fast-skeletal-troponin activator as a means to increase muscle strength by amplifying the response of muscle.",
        "Question: What impact does Tirasemtiv have on muscle?",
              ],
'Subject': ["Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question.",
            "Mention: harming selves male or female prisoners\n Background: The risk of male prisoners harming themselves is ten times that of female prisoners.\n Answer: Between 2004 and 2009; 5-6% of male prisoners and 20-24% of female inmates self-harmed every year",
            "Question: How does gender bias influence self-harm risks of prisoners?",
            "Mention: breast cancer with placental weight\n Background: The risk of breast cancer among parous women increases with placental weight of pregnancies.\n Answer: Placental weight is positively associated with maternal risk of breast cancer",
            "Question: What is the association between placental weight and breast cancer?",

            ],
'Sign_symptom': ["Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question.",
                 "Mention: fine particulate air pollution anxiety\n Background: Exposure to fine particulate air pollution is relate to anxiety prevalence.\n Answer: Exposure to fine particulate matter (PM2.5) was associated with high symptoms of anxiety",
                 "Question: Is there any association between fine particulate air pollution and anxiety level?",
                "Mention: aPKCz tumor glutamine\n Background: aPKCz causes tumour enhancement by affecting glutamine metabolism.\n Answer: PKC\u03b6 represses the expression of two key enzymes of the pathway and phosphorylates PHGDH at key residues to inhibit its enzymatic activity.",
                "Question: What result will aPKCz have when affecting glutamine metabolism in tumor?",
                 ],
}


def convert_to_dict(mention_list):
    new_mention_dict = {}
    for m in mention_list:
        new_mention_dict[str(m['_id'])] = m
    return new_mention_dict


def filter_list(prompts):
    new_prompts = []
    for prompt in prompts:
        if isinstance(prompt, list):
            new_prompts.append(prompt[0])
    return new_prompts


def dstar_query_generation(generator, tokenizer, mention_dict, paths, args):
    """
    Generating query with path of D-STAR
    Args:
        generator: LLaMA generator
        mention_dict:
        paths:

    Returns:

    """
    global logger
    num_request = 0
    results = {}
    output_path = os.path.join(args.output_path, "llama_queries.json")

    for i, path in enumerate(paths):
        # start a new query session with current path
        last_query = None
        for node in path:
            node_info = mention_dict[node]
            mention, context, domain, entity_desc = node_info["mention"], node_info["text"], node_info["domain"], node_info["answer"]
            # pick a demonstration from anchor points for the first node
            if not last_query:
                if domain in DEMO_DICT.keys():
                    _demo = DEMO_DICT[domain]
                else:
                    _demo = random.sample(list(DEMO_DICT.values()), 1)
            else:
                _demo = last_query

            _prompt = "Mention: {}\n Background: {}\n Answer: {}\n Question:".format(mention, context, entity_desc)
            try:
                _prompt = _demo + [_prompt]
                # _prompt = filter_list(_prompt)
                _prompt = " ".join(_prompt)
                _res = generate_query(generator, tokenizer, _prompt)
                _res = filter_results(_res)
                results[str(node)] = _res
            except Exception as e:
                _res = None
                last_query = None
                # Log the error message and exception information
                logger.exception("An error occurs: {}. Continue processing with the current path: {}, node: {}, prompts: {}".format(e, path, node, _demo))
            else:
                last_query = [_demo[0], _prompt, _res]

            if num_request % 10 == 0:
                print("Processed {} queries, saving".format(len(results)))
                print("Processed {}, saving".format(node))

                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)

            num_request += 1

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


def generate_query(generator, tokenizer, prompt):
    results = []
    for word in generate_words(generator, tokenizer, prompt):
        results.append(word)
    return " ".join(results)


def main(args):
    # data
    with open(args.query_path, "r") as f:
        scifact_query_path = json.load(f)

    with open(args.mention_entity, "r") as f:
        mention_entity_dict = json.load(f)

    model_dir = args.llama_path

    generator = ctranslate2.Generator(model_dir, device="cuda")

    # or load on CPU:
    # generator = ctranslate2.Generator(model_dir, device="cpu", intra_threads=6)

    sp = spm.SentencePieceProcessor(os.path.join(model_dir, "tokenizer.model"))

    mention_entity_dict = convert_to_dict(mention_entity_dict)

    dstar_query_generation(generator, sp, mention_entity_dict, scifact_query_path, args)

    logger.info("Finish generating!")


def generate_words(generator, sp, prompt, add_bos=True):
    """

    :param generator: ctr_generator
    :param sp: sentence piece tokenizer
    :param prompt: prompt for the current task
    :param add_bos:
    """
    prompt_tokens = sp.encode(prompt, out_type=str)

    if add_bos:
        prompt_tokens.insert(0, "<s>")

    step_results = generator.generate_tokens(
        prompt_tokens,
        sampling_temperature=0.8,
        sampling_topk=20,
        max_length=512,
        end_token="?"
    )

    output_ids = []

    for step_result in step_results:
        is_new_word = step_result.token.startswith("▁")

        if is_new_word and output_ids:
            yield " " + sp.decode(output_ids)
            output_ids = []

        output_ids.append(step_result.token_id)

    if output_ids:
        yield " " + sp.decode(output_ids)


def print_inline(text):
    print(text, end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Similarity estimation to output k-nn for each claims')
    parser.add_argument('--query-path', required=False, type=str,
                        default="../data/scifact/query_path.jsonl",
                        help='A query path of nodes sampled from the graph')
    parser.add_argument('--mention-entity', required=False, type=str,
                        default="../data/scifact/mention_entity.json",
                        help='output path of the knn neighbors of the current claims')
    parser.add_argument('--output-path', required=False, type=str,
                        default="../data/scifact",
                        help='output path of the knn neighbors of the current claims')
    parser.add_argument('--llama-path', required=True, type=str,
                        help='output path of the knn neighbors of the current claims')
    args = parser.parse_args()
    main(args)