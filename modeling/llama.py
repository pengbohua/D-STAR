from transformers import LlamaTokenizer, LlamaForSequenceClassification, LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import os
import torch
# import accelerate
from eval import move_to_cuda


# model_dir = os.environ['LLAMA']
model_dir = "/home/marvinpeng/llama/7B_meta_ct2"
# model_dir = "/home/marvinpeng/llama/huggingface_llama_13B"
tokenizer = LlamaTokenizer.from_pretrained(model_dir)


prompt1 = "System: Your task will be to generate a Question to the Answer with a Background sentence.\n \
         Mention: flashpoint\n " \
         "Background: a research station on flashpoint to be experimented on by demagol\n " \
         "Answer: flashpoint planet flashpoint was airless planet\n \
         Question: Which flashpoint did demagol have a research station?\n \
         Mention: his girlfriend\n " \
         "Background: 29 felt homesick and often called his girlfriend using star link booth\n " \
         "Answer: lady droid this droid was aspiring actress.\n\
         Question: Who was the girlfriend of 29 contacted with star link booth?\n \
         Mention: steals water\n \
         Background: catch the thief who steals water as a sidedish in this base\n \
         Answer: find water thief is side quest from water guard attacked while staying late.\n \
         Question:"

# prompt2 = "System: Your task will be to generate a Question to the Answer with a Background sentence.\n \
#          Mention: flashpoint\n " \
#          "Background: a research station on flashpoint to be experimented on by demagol\n " \
#          "Answer: flashpoint planet flashpoint was airless planet\n \
#          Question: Which flashpoint did demagol have a research station?\n \
#          Mention: steals water\n \
#          Background: catch the thief who steals water as a sidedish in this base\n \
#          Answer: find water thief is side quest from water guard attacked while staying late.\n \
#          Question:"

stop_list = ["Mention:", "Background:", "Answer:"]
inputs = tokenizer(prompt1, return_tensors="pt")
# stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_list]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze().cuda() for stop_word in stop_list]
inputs = move_to_cuda(inputs)


model = LlamaForCausalLM.from_pretrained(model_dir,
                                         # load_in_8bit=True, device_map='auto'
                                         ).cuda()


class _SentinelTokenStoppingCriteria(StoppingCriteria):

    def __init__(self, sentinel_token_ids: torch.LongTensor,
                 starting_idx: int):
        StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor,
                 _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            # Can't unfold, output is still too tiny. Skip.
            if trimmed_sample.shape[-1] < self.sentinel_token_ids.shape[-1]:
                continue

            for window in trimmed_sample.unfold(
                    0, self.sentinel_token_ids.shape[-1], 1):
                if torch.all(torch.eq(self.sentinel_token_ids, window)):
                    return True
        return False


class StoppingCriteriaWordList(StoppingCriteria):

    def __init__(self, stops, encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]
        # self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


stopping_criteria_list = StoppingCriteriaList([StoppingCriteriaWordList(stops=stop_words_ids)])
generate_ids = model.generate(inputs.input_ids,
                              max_new_tokens=200,
                              temperature=0.7, 
                              # top_p=1.0,
                              # top_k=0,
                              # typical_p=1.0,
                              # repetition_penalty=1.05,
                              # stopping_criteria=stopping_criteria_list,
                              early_stopping=True)

print(tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])