from argparse import ArgumentParser

parser = ArgumentParser(description='test arguments')
parser.add_argument('--task', type=str, default='in_domain',
                    help='Perform cross domain entity linking or in domain entity linking')
parser.add_argument("--document-files", nargs="+", default=None,
                    help="Path to train documents json file.")
parser.add_argument("--test-mentions-file", default='/home/marvinpeng/datasets/data_zip/Fandomwiki/mentions/test.json',
                    type=str,
                    help="Path to test mentions json file.")
parser.add_argument("--test-tfidf-candidates-file",
                    default='/home/marvinpeng/datasets/data_zip/Fandomwiki/tfidf_candidates/test_tfidfs.json', type=str,
                    help="Path to test candidates file retrieved by BM25.")
parser.add_argument('--model-type', type=str, default='bi_encoder', help='bi_encoder or cross_encoder')
parser.add_argument('--eval-model-path', type=str, default='checkpoints/cross_cross_domain/model_best.ckpt')
parser.add_argument("--use-tf-idf-negatives", action="store_true",
                    help="Use tf-idf as hard negatives in contrastive learning.")
parser.add_argument("--use-mention-negatives", action="store_true",
                    help="Use in-batch mention negatives as hard negatives in contrastive learning.")
parser.add_argument("--max-seq-length", default=64, type=int, help="Maximum sequence length.")
parser.add_argument("--num-candidates", default=32, type=int, help="Number of tfidf candidates (0-63).")
args = parser.parse_args()