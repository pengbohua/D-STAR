import argparse
from dataclasses import dataclass, field


@dataclass
class TrainingArguments:
    weight_decay: float = field(default=1e-4,
                            metadata={"help": "weight decay parameter for optimization"}
                            )
    grad_clip: float = field(default=10,
                            metadata={"help": "magnitude for gradient clipping"}
                            )
    epochs: int = field(default=3,
                            metadata={"help": "number of training epochs"}
                            )
    num_cand: int = field(default=64,
                            metadata={"help": "number of negative samples"}
                            )
    warmup: int = field(default=500,
                        metadata={"help": "warmup steps"})
    use_amp: bool = field(default=False,
                        metadata={"help": "use mixed precision"})
    train_batch_size: int = field(default=2,
                        metadata={"help": "train batch size"})
    eval_batch_size: int = field(default=2,
                        metadata={"help": "eval batch size"})
    eval_every_n_steps: int = field(default=1000,
                        metadata={"help": "eval every n steps"})
    log_every_n_steps: int = field(default=100,
                        metadata={"help": "log every n steps"})
    max_weights_to_keep: int = field(default=3,
                                     metadata={"help": "max number of weight file to keep"})


def get_args():
    parser = argparse.ArgumentParser("zero shot entity linker")

    parser.add_argument("--pretrained-model-path", default='bert-base-uncased', type=str,
                        help="Path to pretrained transformers.")
    parser.add_argument("--eval-model-path", default='checkpoints', type=str,
                        help="Path to pretrained transformers.")
    parser.add_argument("--document-files", nargs="+", default=None,
                        help="Path to train documents json file.")
    parser.add_argument("--train-mentions-file", default=None, type=str,
                        help="Path to mentions json file.")
    parser.add_argument("--eval-mentions-file", default=None, type=str,
                        help="Path to mentions json file.")
    parser.add_argument("--train-tfidf-candidates-file", default='tfidf_candidates/train_tfidfs.json', type=str,
                        help="Path to TFIDF candidates file.")
    parser.add_argument("--eval-tfidf-candidates-file", default='tfidf_candidates/test_tfidfs.json', type=str,
                        help="Path to TFIDF candidates file.")
    parser.add_argument(
        "--split-by-domain", default=False, type=bool,
        help="Split output data file by domain.")
    parser.add_argument("--learning-rate", default=1e-3, type=float,
                        help="learning rate for optimization")
    parser.add_argument("--weight-decay", default=1e-4, type=float,
                        help="weight decay for optimization")
    parser.add_argument("--epochs", default=1, type=int,
                        help="weight decay for optimization")
    parser.add_argument("--train-batch-size", default=16, type=int,
                        help="train batch size")
    parser.add_argument("--eval-batch-size", default=128, type=int,
                        help="train batch size")
    parser.add_argument("--num-workers", default=12, type=int,
                        help="number of workers for data loading")
    parser.add_argument("--margin", default=0.00, type=float,
                        help="additive margin for contrastive learning")
    parser.add_argument("--model-type", default='contrastive', type=str,
                        help="contrastive crossencoder")
    parser.add_argument("--bitfit", action='store_true', help="Use bitfit tuning for entity linking")
    parser.add_argument("--adapter", action='store_true', help="Use adapter tuning for entity linking")
    parser.add_argument("--lora", action='store_true', help="Use low rank tuning for entity linking")
    parser.add_argument("--prefix", action='store_true', help="Use prefix tuning for entity linking")
    parser.add_argument("--prefix-length", default=180, type=int, help="Prefix length for entity linking")
    parser.add_argument("--max-seq-length", default=32, type=int, help="Maximum sequence length.")

    parser.add_argument("--num-candidates", default=64, type=int, help="Number of tfidf candidates (0-63).")

    parser.add_argument("--random-seed", default=12345, type=int, help="Random seed for data generation.")

    parser.add_argument("--use-tf-idf-negatives", action="store_true", help="Use tf-idf as hard negatives in contrastive learning.")

    parser.add_argument("--use-mention-negatives", action="store_true", help="Use in-batch mention negatives as hard negatives in contrastive learning.")

    args = parser.parse_args()
    return args


args = get_args()