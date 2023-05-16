import json
import time
import os
import openai
from openai.error import RateLimitError, InvalidRequestError
from tqdm import tqdm
import logging
import random


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler to write the log to a file
log_file = 'colbert/query_generation/log.txt'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter for the log message
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger object
logger.addHandler(file_handler)
logger.info('Starting query generation...')


key1 = "KEY1"
key2 = "KEY2"
key1 = "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"
key2 = "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"
key_set = set()
key_set.add(key1)
key_set.add(key2)
openai.api_key = key1

# Unbiased (not just supportive but also contradict) demonstrations provided by medical experts
DEMO_DICT = {
"Disease_disorder": [{"role": "system", "content": "Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question."},
        {"role": "user", "content": "Mention: chronic kidney disease\n Background: 40mg/day dosage of folic acid does not affect chronic kidney disease (CKD) progression\n Answer:Treatment with high doses of folic acid did not reduce the incidence of vascular disease in patients with advanced chronic kidney disease."},
        {"role": "assistant", "content": "What is the impact of folic acid on the progression of chronic kidney disease?"},
        {"role": "user", "content": "Mention: main hiv\n Background: Having a main partner worsens HIV outcomes.\n Answer: In an analysis stratified by previous antiretroviral therapy and clinical stage when starting HAART (US Centers for Disease Control), the adjusted hazard ratio for progression to AIDS was 0.79 for participants with a stable partnership compared with those without."},
        {"role": "assistant", "content": "What kind of HIV outcomes will having a main partner lead to from some government statistics?"},
                     ],
'Biological_structure': [{"role": "system", "content": "Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question."},
        {"role": "user", "content": "Mention: liver methadone\n Background: 32% of liver transplantation programs required patients to discontinue methadone treatment in 2001.\n Answer: Policies requiring discontinuation of methadone in 32% of all programs contradict the evidence base for efficacy of long-term replacement therapies."},
        {"role": "assistant", "content": "Question: What impact do policies which require discontinuation of methadone treatment have on the efficacy of liver transplant programs?"},
        {"role": "user", "content": "Mention: breast cancer genetic factors\n Background: Breast cancer development is determined exclusively by genetic factors.\n Answer: Risks of breast cancer associated with low-penetrance susceptibility polymorphisms do not vary significantly with these ten established environmental risk factors."},
    {"role": "assistant", "content": "Question: Is there a correlation between breast cancer and genetic factors?"},],

'Diagnostic_procedure': [{"role": "system", "content": "Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question."},
        {"role": "user", "content": "Mention: bias in the phage genome locations\n Background: A strong bias in the phage genome locations where the spacers were derived has been observed in many CRISPR subtypes that confer the immunity to phage\n Answer: We detect a strong and reproducible bias in the phage genome locations from which spacers derive"},
        {"role": "assistant", "content": "Question: Which level of bias is observed in the phage genome locations?"},

        {"role": "user",
        "content": "Mention: TDP-43 respiratory proteins neuronal loss\n Background: Blocking the interaction between TDP-43 and respiratory complex I proteins leads to increased neuronal loss.\n Answer: The suppression of TDP-43 mitochondrial localization abolishes mutant TDP-43-induced neuronal loss"},
        {"role": "assistant",
        "content": "Question: What is the impact of blocking TDP-43 and respiratory proteins on neuronal loss?"},
                         ],
'Detailed_description': [{"role": "system", "content": "Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question."},
        {"role": "user", "content": "Mention: chronic tension type headache\n Background: Amitriptyline is an effective treatment for chronic tension-type headaches.\n Answer: Tricyclic antidepressant medication produced larger reductions in headache activity, analgesic medication use, and headache-related disability than placebo"},
        {"role": "assistant", "content": "Question: What is the efficacy of antidepressants for treating chronic tension-type headaches?"},
        {"role": "user",
        "content": "Mention: infection rate transplantation of mesenchymal stem cells induction therapy\n Background: Autologous transplantation of mesenchymal stem cells causes a higher rate of opportunistic infections than induction therapy with anti-interleukin-2 receptor antibodies.\n Answer: the use of autologous MSCs compared with antibody induction therapy resulted in decreased risk of infection"},
        {"role": "assistant",
        "content": "Question: Which will cause a higher infection rate, transplantation of mesenchymal stem cells or induction therapy?"},
                         ],

'Medication':[{"role": "system", "content": "Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question."},
        {"role": "user", "content": "Mention: antidepressants\n Background: Antidepressants reduce the severity of migraines\n Answer: Tricyclics significantly reduced the number of days with tension-type headache and number of headache attacks from migraine than placebo"},
        {"role": "assistant", "content": "Question: How effective are antidepressants for treating migraines?"},
        {"role": "assistant", "content": "Mention: Tirasemtiv muscle\n Background: Tirasemtiv has no effect on fast-twitch muscle.\n Answer: We developed a small-molecule fast-skeletal-troponin activator as a means to increase muscle strength by amplifying the response of muscle. "},
        {"role": "assistant", "content": "Question: What impact does Tirasemtiv have on muscle?"},
              ],    
'Subject': [{"role": "system", "content": "Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question."},
            {"role": "user", "content": "Mention: harming selves male or female prisoners\n Background: The risk of male prisoners harming themselves is ten times that of female prisoners.\n Answer: Between 2004 and 2009; 5-6% of male prisoners and 20-24% of female inmates self-harmed every year"},
            {"role": "assistant", "content": "Question: How does gender bias influence self-harm risks of prisoners?"},

            {"role": "user", "content": "Mention: breast cancer with placental weight\n Background: The risk of breast cancer among parous women increases with placental weight of pregnancies.\n Answer: Placental weight is positively associated with maternal risk of breast cancer"},
            {"role": "assistant", "content": "Question: What is the association between placental weight and breast cancer?"},

            ],
'Sign_symptom': [{"role": "system", "content": "Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question."},
                 {"role": "user",
                  "content": "Mention: fine particulate air pollution anxiety\n Background: Exposure to fine particulate air pollution is relate to anxiety prevalence.\n Answer: Exposure to fine particulate matter (PM2.5) was associated with high symptoms of anxiety"},
                 {"role": "assistant",
                  "content": "Question: Is there any association between fine particulate air pollution and anxiety level?"},

                 {"role": "user",
                  "content": "Mention: aPKCz tumor glutamine\n Background: aPKCz causes tumour enhancement by affecting glutamine metabolism.\n Answer: PKC\u03b6 represses the expression of two key enzymes of the pathway and phosphorylates PHGDH at key residues to inhibit its enzymatic activity."},
                 {"role": "assistant",
                  "content": "Question: What result will aPKCz have when affecting glutamine metabolism in tumor?"},
                 ],
}


def assemble_query(mention, context, entity_desc):
    """
    Add current query to demonstrations as prompts
    Args:
        demonstrations:
        mention:
        context:
        entity_desc:

    Returns:

    """
    # pseudo_name = random.randint(1, 99999)
    query_dict = {"role": "user",
                   "content": "Mention: {}\n Background: {}\n Answer: {}".format(mention, context, entity_desc),
                   # "name": "{}".format(pseudo_name)
                   }

    return query_dict


def _send_query(mention_id, prompt):

    response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  max_tokens=16,
  temperature=0.7,
  stop=["?", ",", "and"],
  frequency_penalty=0,
  presence_penalty=0,
  messages=prompt
)

    try:
        result = response.choices[0].message.content
        return {"mention_id": mention_id, "result": result, "prompt": prompt}, response
    except:
        logging.info("No message or content for current query")
        return {"mention_id": mention_id, "result": None, "prompt": prompt}, response


def convert_to_dict(mention_list):
    new_mention_dict = {}
    for m in mention_list:
        new_mention_dict[str(m['_id'])] = m
    return new_mention_dict


def naive_query_generation(samples):
    """
    Generate queries with fixed demonstration as anchor points.
    Will raise max context exceeding error (InvalidRequestError) because GPT3.5 take similar requests as dialogue
    Args:
        samples:

    Returns:

    """
    results = {}
    for i, sample_dict in tqdm(enumerate(samples[1:])):
        mid, mention, context, corpus, entity_desc = sample_dict["mention_id"], sample_dict["mention"], sample_dict["context"], sample_dict["corpus"], sample_dict["entity_desc"]
        _domain = corpus
        _demo = DEMO_DICT[_domain]
        _query = assemble_query(mention, context, entity_desc)
        try:
            _res, response = _send_query(mid, _demo+[_query])
            results[mid] = _res
        except RateLimitError:
            time.sleep(20)
            temp_key = openai.api_key
            openai.api_key = random.sample(key_set - {temp_key}, 1)
            _res, response = _send_query(mid, _demo+[_query])
            openai.api_key = temp_key
            results[mid] = _res
        except InvalidRequestError:
            logger.error("Maximum context exceeded: {}".format(sample_dict))
            openai.api_key = key2
            _demo = _demo[:3]   # remove current path when max context error occurs
            prompts = assemble_query(_demo, mention, context, entity_desc)
            _res, response = _send_query(_demo, prompts)
            results[mid] = _res
        except Exception as e:
            # Log the error message and exception information
            logger.exception("An error occurred: {}\n {}".format(e, sample_dict))

        time.sleep(22)

        if i % 10 == 0:
            print("Processed {} queries, saving".format(i+1))
            with open("./test_queries1.json", "w") as f:
                json.dump(results, f, indent=4)

    with open("./test_queries.json", "w") as f:
        json.dump(results, f, indent=4)


def dstar_query_generation(mention_dict, paths):
    """
    Generating query with path of D-STAR
    Args:
        mention_dict:
        paths:

    Returns:

    """
    global logger
    num_request = 0
    results = {}
    failed_cases = {}

    for i, path in tqdm(enumerate(paths)):
        # start a new dialogue session with current path
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

            _query = assemble_query(mention, context, entity_desc)
            try:
                _res, response = _send_query(node, _demo+[_query])
                results[node] = _res
            except RateLimitError:
                time.sleep(20)
                temp_key1 = openai.api_key
                openai.api_key = random.sample(key_set - {temp_key1}, 1)[0]
                try:
                    _res, response = _send_query(node, _demo+[_query])
                except RateLimitError:
                    time.sleep(15)
                    openai.api_key = random.sample(key_set - {temp_key1, openai.api_key}, 1)[0]
                    _res, response = _send_query(node, _demo+[_query])
                openai.api_key = key1
                results[node] = _res
            except InvalidRequestError:
                time.sleep(15)
                logger.error("Maximum context exceeded: node:{}, prompt: {}".format(node, _demo+[_query]))
                temp_key = openai.api_key
                openai.api_key = random.sample(key_set - {temp_key}, 1)[0]
                if domain in DEMO_DICT.keys():
                    _demo = DEMO_DICT[domain]  # remove current path when max context error occurs
                else:
                    _demo = random.sample(list(DEMO_DICT.values()), 1)
                _query = assemble_query(mention, context, entity_desc)
                try:
                    _res, response = _send_query(node, _demo+[_query])
                except Exception as e:
                    failed_cases[i] = (path, node)
                    logger.error("Request fails: {}. Discarding the current path: {}, node: {}, prompt: {}".format(e, path, node, _demo))
                    break
                else:
                    results[node] = _res
                    last_query = [_demo[0], _query, {"role": "assistant", "content": _res['result']}]
                openai.api_key = temp_key
            except Exception as e:
                _res = None
                last_query = None
                # Log the error message and exception information
                logger.exception("An error occurs: {}. Continue processing with the current path: {}, node: {}, prompts: {}".format(e, path, node, _demo))
            else:
                last_query = [_demo[0], _query, {"role": "assistant", "content": _res['result']}]

            if num_request % 3 == 0:
                print("Processed {} queries, saving".format(i+1))
                with open("colbert/query_generation/queries_temp.json", "w") as f:
                    json.dump(results, f, indent=4)

            num_request += 1
            time.sleep(22)

    with open("colbert/query_generation/queries.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    with open("../data/scifact/query_path.jsonl", "r") as f:
        scifact_query_path = json.load(f)

    with open("../data/scifact/mention_entity.json", "r") as f:
        mention_entity_dict = json.load(f)
    total_nodes = 0
    print("mentions", len(mention_entity_dict))

    for path in scifact_query_path:
        total_nodes += len(path)

    mention_entity_dict = convert_to_dict(mention_entity_dict)
    dstar_query_generation(mention_entity_dict, scifact_query_path)

