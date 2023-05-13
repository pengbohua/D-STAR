import openai
import logging
import json
import time
import os
import openai
from openai.error import RateLimitError, InvalidRequestError
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler to write the log to a file
log_file = 'data/In-Zeshel/text_style_transfer/logs/log.txt'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter for the log message
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger object
logger.addHandler(file_handler)
logger.info('Starting query generation...')


key1 = os.getenv("OPENAIKEY1")
key2 = os.getenv("OPENAIKEY2")
key3 = os.getenv("OPENAIKEY3")
openai.api_key = key1

SUPER_DOMAIN = {"drama": {"coronation_street", "muppets"},
'sports': {"american_football", "ice_hockey", "pro_wrestling"},
'scifi': {"starwars", "doctor_who", "star_trek",  "lego", "final_fantasy", "yugioh",},
'fantasy': {"fallout", "world_of_warcraft", "elder_scrolls", "forgotten_realms", "military"},
}

DEMO_DICT = {
"drama": [{"role": "system", "content": "Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question."},
        {"role": "user", "content": "flat\n Background: Dev drags Tara back to his flat demanding explanation\n Answer: 9 Victoria court is apartment in Victoria"},
        {"role": "assistant", "content": "Question: Where is the flat that Dev drags Tara back?"},
        {"role": "user", "content": "Mention: chicken lake\n Background: artist sketches include chicken lake with gonzo and chickens\n Answer: swan lake is ballet by Pyotr Ilyich Tchaikovsky"},
       {"role": "assistant", "content": "Question: What art does chicken lake sarcastically reflect?"},],
'sports': [{"role": "system", "content": "Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question."},
        {"role": "user", "content": "2006 - 07 season\n Background: league sphl for 2006 07 season playing in 52 regular\n Answer: 2006 07 southern professional hockey league season 2006"},
        {"role": "assistant", "content": "Question: Which season did league sphl play in 52 regular?"},
        {"role": "user", "content": "Mention: 4 - 3 defence\n Background: defensive end in standard 4 - 3 defence or as fourth linebacker\n Answer: American football strategy concerns deployment of defensive and special teams players"},
       {"role": "assistant", "content": "Question: Which category does 4 - 3 defence belong to in American football?"},],
'scifi': [{"role": "system", "content": "Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question."},
        {"role": "user", "content": "Mention: flashpoint\n Background: a research station on flashpoint to be experimented on by demagol\n Answer: flashpoint planet flashpoint was airless planet"},
        {"role": "assistant", "content": "Question: Which flashpoint did demagol have a research station?"},
        {"role": "user", "content": "Mention: his girlfriend\n Background: 29 felt homesick and often called his girlfriend using star link booth\n Answer: lady droid this droid was aspiring actress"},
       {"role": "assistant", "content": "Question: Who was the girlfriend of 29 contacted with star link booth?"},],
'fantasy': [{"role": "system", "content": "Your task will be to generate a Question to the Answer with a Background sentence. Make sure not to leak the Answer directly to the Question."},
        {"role": "user", "content": "Mention: steals water\n Background: catch the thief who steals water as a sidedish in this base\n Answer: find water thief is side quest from water guard attacked while staying late"},
        {"role": "assistant", "content": "Question: Which flashpoint did demagol have a research station?"},
        {"role": "user", "content": "Mention: his girlfriend\n Background: 29 felt homesick and often called his girlfriend using star link booth\n Answer: lady droid this droid was aspiring actress"},
       {"role": "assistant", "content": "Question: Who was the girlfriend of 29 contacted with star link booth?"},],
}


def _get_query(mention_id, demonstrations, mention, context, entity_desc):
    prompt_dict = {"role": "user", "content":"Mention: {}\n Background: {}\n Answer: {}".format(mention, context, entity_desc)}
    demonstrations.append(prompt_dict)
    response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  max_tokens=20,
  temperature=0.7,
  stop=["?", ",", "and"],
  frequency_penalty=0,
  presence_penalty=0,
  messages=demonstrations
)

    try:
        result = response.choices[0].message.content
        return {"mention_id": mention_id, "result": result, "prompt": prompt_dict}, response
    except:
        logging.info("No message or content for current query")
        return {"mention_id": mention_id, "result": None, "prompt": prompt_dict}, response


if __name__ == "__main__":
    with open("data/In-Zeshel/test_contextualized_amb_substring_samples.json", "r") as f:
        amb_substring_samples = json.load(f)

    with open("data/In-Zeshel/test_contextualized_less_overlap_samples.json", "r") as f:
        less_overlap_samples = json.load(f)

    samples = less_overlap_samples + amb_substring_samples

    results = []
    for i, sample_dict in tqdm(enumerate(samples[1:])):
        mid, mention, context, corpus, entity_desc = sample_dict["mention_id"], sample_dict["mention"], sample_dict["context"], sample_dict["corpus"], sample_dict["entity_desc"]
        for k, domains in SUPER_DOMAIN.items():
            if corpus in domains:
                _demo = DEMO_DICT[k]       
        try:
            _res, response = _get_query(mid, _demo, mention, context, entity_desc)
            results.append(_res)
        except RateLimitError:
            time.sleep(5)
            openai.api_key = key2
            _res, response = _get_query(mid, _demo, mention, context, entity_desc)
            openai.api_key = key1
            results.append(_res)
        except InvalidRequestError:
            logger.error("Maximum context exceeded: {}".format(sample_dict))
            openai.api_key = key2
            _demo = _demo[:3]
            _res, response = _get_query(mid, _demo, mention, context, entity_desc)
        except Exception as e:
            # Log the error message and exception information
            logger.exception("An error occurred: {}\n {}".format(e, sample_dict))

        time.sleep(10)

        if i % 10 == 0:
            print("Processed {} queries, saving".format(i+1))
            with open("./test_queries.json", "w") as f:
                json.dump(results, f, indent=4)    

    with open("./test_queries.json", "w") as f:
        json.dump(results, f, indent=4)
