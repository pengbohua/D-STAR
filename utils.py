import logging
import json
import time
import csv
import tqdm

def load_candidates(input_dir):
    documents = {}
    with open(input_dir, "r", encoding="utf-8") as reader:
        doc_dicts = json.load(reader)

    for doc in doc_dicts:
        documents[doc['mention_id']] = doc["tfidf_candidates"]  # mention_id to candidates

    return documents

def load_documents(input_dir):
    '''
    load all documents of Zeshel into a large dictionary
    '''
    documents = {}
    with open(input_dir, "r", encoding="utf-8") as reader:
        domain = input_dir.split('/')[-1].replace('_', ' ').replace('.json', '')
        while True:
            line = reader.readline()
            line = line.strip()
            if not line:
                break
            line = json.loads(line)
            line['metadata'] = domain
            documents[line['document_id']] = line
    logger.info("Loading {} documents from {}".format(len(documents), input_dir))
    return documents


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger

logger = _setup_logger()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def convert_json_tsv(j_file, tsv_file):
    new_data = {}
    with open(j_file, 'r') as jIn:
        for line in jIn:
            data_dict = json.loads(line)
            new_data[data_dict['_id']] = data_dict['text']
    # save queries to queries.tsv _id\t query
    print("Preprocessing {} json and Saving to {} ...".format(len(new_data), tsv_file))
    with open(tsv_file, 'w') as fIn:
        writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for qid, query in tqdm.tqdm(new_data.items()):
            writer.writerow([qid, query])